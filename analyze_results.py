import pandas as pd

df = pd.read_csv('results/evaluation_results_bciiv2a_20260308_145404.csv')
fb_df = df[df['algorithm'] == 'FilterBankTangentSpace+SVM']

print('各受试者平均准确率:')
print(fb_df.groupby('subject')['accuracy'].mean().round(4))

print(f'\n总体平均: {fb_df["accuracy"].mean():.4f}')
print(f'最高准确率: {fb_df["accuracy"].max():.4f}')
print(f'最低准确率: {fb_df["accuracy"].min():.4f}')