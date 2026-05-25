# ==========================================
# 
# 功能: 模型训练 (嵌套 OOF + 健康人最终模型, 可用于非健康人预测)
# ==========================================

# ==========================================
# 变更说明 (与旧代码相比):
# 1. 引入嵌套交叉验证 (Nested OOF) 以确保训练集内部预测结果的严格性。
#    - 外层 5 折生成 OOF
#    - 内层 3 折用于 Optuna 超参数调优
# 2. 最终模型训练独立使用全量健康人数据，并保存为 LGBM_<organ>_final.pkl，确保外部非健康人预测时数据完全隔离。
# 3. OOF 预测结果文件名和最终生成的模型文件名保持统一管理，便于调用。
# 4. 明确区分训练阶段和外部验证阶段，非健康人数据不参与任何训练和调参，确保数据隔离。
# 5. 增强了打印和日志信息，标识嵌套 OOF 进度和最终模型保存情况。
# ==========================================

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import joblib
import os
import warnings

# 基础配置
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# === 路径配置 ===
AGE_COL = 'age_at_assessment'
FEATURE_FILE = '/Users/zkf/Desktop/新项目/预处理后的原始数据/HPA/features_union.csv'
MODEL_DIR = '/Users/zkf/Desktop/test/file/TrainModel'
OUTPUT_DIR = '/Users/zkf/Desktop/test/file'
N_TRIALS = 30  # 可以根据时间调整

# --- 核心函数 ---
def run_nested_oof_training(df_train):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    feat_df = pd.read_csv(FEATURE_FILE)
    feat_df['meaning'] = feat_df['meaning'].astype(str)
    organs = feat_df['organ'].unique()

    oof_file_name = os.path.join(OUTPUT_DIR, 'train_predictions_all_nested_oof.csv')

    df_train_oof = df_train[['eid', AGE_COL]].copy()

    print(f"🏭 [Train] 开始构建器官时钟 (Nested OOF)...")

    for organ in organs:
        print(f"\n>> 正在处理器官: {organ}")
        organ_genes = feat_df[feat_df['organ'] == organ]['meaning'].tolist()
        features = [c for c in organ_genes if c in df_train.columns]
        missing_features = [c for c in organ_genes if c not in df_train.columns]
        if missing_features:
            print(f"   ℹ️  该器官缺失 {len(missing_features)} 个特征 (训练集中不存在)")
        if len(features) < 5:
            print(f"   ⚠️ [跳过] 有效特征太少 ({len(features)})")
            continue

        X_df = df_train[features]
        X_numpy = X_df.values
        y = df_train[AGE_COL].values

        # --- 嵌套 OOF ---
        oof_preds = np.zeros(X_numpy.shape[0])
        outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for outer_train_idx, outer_val_idx in outer_kf.split(X_numpy):
            X_outer_train, y_outer_train = X_numpy[outer_train_idx], y[outer_train_idx]
            X_outer_val = X_numpy[outer_val_idx]

            # 内层 CV + Optuna 寻参
            def objective(trial):
                param = {
                    'objective': 'regression',
                    'metric': 'mae',
                    'verbosity': -1,
                    'boosting_type': 'gbdt',
                    'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'n_jobs': 1
                }
                inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
                maes = []
                for t_idx, v_idx in inner_kf.split(X_outer_train):
                    model = lgb.LGBMRegressor(**param)
                    model.fit(X_outer_train[t_idx], y_outer_train[t_idx])
                    preds = model.predict(X_outer_train[v_idx])
                    maes.append(mean_absolute_error(y_outer_train[v_idx], preds))
                return np.mean(maes)

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=N_TRIALS)
            best_params = study.best_params

            final_params = best_params.copy()
            final_params.update({'n_jobs': -1, 'random_state': 42})
            model = lgb.LGBMRegressor(**final_params)
            model.fit(X_outer_train, y_outer_train)
            oof_preds[outer_val_idx] = model.predict(X_outer_val)

        df_train_oof[f'{organ}_Age'] = oof_preds
        print(f"   ✅ 完成 OOF 预测")

        # --- 用全部健康人训练最终模型 ---
        final_model_name = os.path.join(MODEL_DIR, f'LGBM_{organ}_final.pkl')
        print(f"   Step: 使用全量健康人训练最终模型: {final_model_name}")
        def final_objective(trial):
            param = {
                'objective': 'regression',
                'metric': 'mae',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'n_jobs': -1
            }
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            maes = []
            for t_idx, v_idx in cv.split(X_numpy):
                m = lgb.LGBMRegressor(**param)
                m.fit(X_numpy[t_idx], y[t_idx])
                preds = m.predict(X_numpy[v_idx])
                maes.append(mean_absolute_error(y[v_idx], preds))
            return np.mean(maes)

        study_final = optuna.create_study(direction='minimize')
        study_final.optimize(final_objective, n_trials=N_TRIALS)
        best_final_params = study_final.best_params
        best_final_params.update({'n_jobs': -1, 'random_state': 42})

        final_model = lgb.LGBMRegressor(**best_final_params)
        final_model.fit(X_df, y)
        joblib.dump(final_model, final_model_name)
        print(f"   ✅ 最终模型已保存")

    df_train_oof.to_csv(oof_file_name, index=False)
    print(f"\n[Train] 所有器官训练完成, OOF 文件已保存: {oof_file_name}")
    return df_train_oof

# --- 执行逻辑 ---
if __name__ == '__main__':
    print("正在读取训练数据...")
    df_train_proteomic = pd.read_csv('/Users/zkf/Desktop/新项目/预处理后的原始数据/蛋白质训练测试数据/all_train.csv')
    new_columns = [col.upper() if col != 'eid' else col for col in df_train_proteomic.columns]
    df_train_proteomic.columns = new_columns

    df_demography = pd.read_csv('/Users/zkf/Desktop/新项目/预处理后的原始数据/最终人口学数据/demography.csv', usecols=['eid', 'age_at_assessment'])
    df_train = pd.merge(df_train_proteomic, df_demography, on='eid', how='inner')

    df_oof_result = run_nested_oof_training(df_train)
