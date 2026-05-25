# ==========================================
# 文件名: 1.3-整个人群Test.py
# 功能: 测试集预测值生成 (不计算Gap)
# ==========================================

import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# === 路径配置 ===
AGE_COL = 'age_at_assessment'
FEATURE_FILE = '/Users/zkf/Desktop/新项目/预处理后的原始数据/HPA/features_union.csv'
MODEL_DIR = '/Users/zkf/Desktop/test/file/TrainModel'
OUTPUT_DIR = '/Users/zkf/Desktop/test/file'


def predict_organ_ages(df_test):
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError("❌ 模型文件夹不存在，请先运行 Train！")

    # 重新读取特征表以确定每个器官需要哪些特征
    feat_df = pd.read_csv(FEATURE_FILE)
    organs = feat_df['organ'].unique()

    df_res = df_test[['eid', AGE_COL]].copy()

    print(f"🚀 [Test] 开始预测 {len(organs)} 个器官...")

    for organ in organs:
        model_path = os.path.join(MODEL_DIR, f'LGBM_{organ}.pkl')
        if not os.path.exists(model_path):
            print(f"❌ {organ}: 模型丢失，跳过")
            continue

        # 获取特征
        organ_genes = feat_df[feat_df['organ'] == organ]['meaning'].tolist()
        organ_genes = [c for c in organ_genes if c in df_test.columns]
        # 注意：这里假设 Test 数据集包含所有 Train 里的特征
        # 如果 Test 可能缺失某些列，需要补 0 处理
        missing_cols = [c for c in organ_genes if c not in df_test.columns]
        if missing_cols:
            print(f"⚠️ {organ}: 缺少 {len(missing_cols)} 个特征，填充 0")
            for c in missing_cols:
                df_test[c] = 0

        features = [c for c in organ_genes if c in df_test.columns]  # 确保顺序一致
        X_test = df_test[features].copy()

        # 加载模型 & 预测
        model = joblib.load(model_path)
        raw_pred = model.predict(X_test)

        # 仅保存原始预测值
        df_res[f'{organ}_Age'] = raw_pred
        print(f"✅ {organ} Predicted")

    print(f"✅ [Test] 预测完成。")
    return df_res


# --- 执行逻辑 ---
df_test_proteomic = pd.read_csv('/Users/zkf/Desktop/新项目/预处理后的原始数据/蛋白质训练测试数据/all_test.csv')
new_columns = [col.upper() if col != 'eid' else col for col in df_test_proteomic.columns]
df_test_proteomic.columns = new_columns
df_demography = pd.read_csv('/Users/zkf/Desktop/新项目/预处理后的原始数据/最终人口学数据/demography.csv',
                            usecols=['eid', 'age_at_assessment'])
df_test = pd.merge(df_test_proteomic, df_demography, on='eid', how='inner')

# 运行并保存
df_predictions = predict_organ_ages(df_test)
# 输出文件名为 test_predictions_raw.csv
df_predictions.to_csv(os.path.join(OUTPUT_DIR, 'test_predictions_all.csv'), index=False)

