#%%
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns', None)
# %%
# Чтение файла с тегами.
df = pd.read_excel(r'E:\Projects\portfolio\ad_creative_performance_analyzer\materials\tagging.xlsx')

# %%

cols_drop = ['adtext_text','adtext_eng','headline_text','headline_eng','description_text','image_text',
               'image_eng_text','image_clip','image_clip_keywords']

#%%

def analyze_ctr_feature_importance(df, cols_drop, top_n=10, plot=True):
    """
    Основная функция для оценки степени влияния признаков креатива на CTR.
    
    Параметры:
    ----------
    df : pd.DataFrame
        Исходный датасет.
    top_n : int
        Количество топовых признаков для вывода на графике.
    plot : bool
        Строить ли график результатов.
    drop_cols : str
        Список, колонок, которые не нужно учитывать
        
    Возвращает:
    -----------
    pd.DataFrame со степенями влияния признаков в процентах.
    """
#%%
df_work = df.copy()

# --- 1. Очистка от сырых текстов и системных полей ---
cols_to_drop = [col for col in cols_drop if col in df_work.columns]
df_work = df_work.drop(columns = cols_drop)

# Адаптация под размер выборки (для 15 баннеров и для 1000+)
n_samples = len(df_work)
min_freq = 1 if n_samples < 30 else 3

# ==== ОБРАБОТКА МАССИВОВ ====
# Поиск множественных строк (псевдо списков)  и перевод их в реальные списки

def is_text_array(val):
    # Функция, которая проверяет, является ли текст замаскированным списком
    if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
        try:
            # Пробуем безопасно прочитать структуру текста
            parsed = ast.literal_eval(val)
            return isinstance(parsed, list)
        except (ValueError, SyntaxError):
            return False
    return False

# Ищем все колонки, где есть такие текстовые списки
text_array_cols = [
    col for col in df_work.columns 
    if df_work[col].dropna().apply(is_text_array).any()
    ]

# Переводим их в настоящие списки Python
# for col in text_array_cols:
#     # 1. Безопасно превращаем строки "['a', 'b']" в реальные списки Python ['a', 'b']
#     df_work[col] = df_work[col].apply(
#         lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
#     )

# ==== ПОИСК ОБЫЧНЫХ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ ====

# Отбираем все колонки с типом данных object или category
potential_cat_cols = df_work.select_dtypes(include=['object', 'category']).columns

# Исключаем из них те, которые уже сохранены в text_array_cols
categorical_cols = [col for col in potential_cat_cols if col not in text_array_cols]

print(text_array_cols)
print(categorical_cols)

#%%

# Адаптация под размер выборки (для 15 баннеров и для 1000+)
n_samples = len(df_work)
min_freq = 1 if n_samples < 30 else 3

# Словарь для маппинга созданных колонок к родительской группе
feature_groups = {}

for col in text_array_cols:
    # Безопасно превращаем строки "['a', 'b']" в реальные списки Python ['a', 'b']
    df_work[col] = df_work[col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') 
        else ([] if not isinstance(x, list) else x)
    )
    
    mlb = MultiLabelBinarizer()
    binary_matrix = mlb.fit_transform(df_work[col])
    
    # Отсеиваем слишком редкие значения
    class_counts = binary_matrix.sum(axis=0)
    valid_classes_idx = [i for i, count in enumerate(class_counts) if count >= min_freq]
    
    valid_classes = [mlb.classes_[i] for i in valid_classes_idx]
    valid_matrix = binary_matrix[:, valid_classes_idx]
    
    col_names = [f"{col}_{cls}" for cls in valid_classes]
    df_bin = pd.DataFrame(valid_matrix, columns=col_names, index=df_work.index)
    
    # Привязываем бинарные колонки к исходной группе
    for cn in col_names:
        feature_groups[cn] = col
        
    df_work = pd.concat([df_work, df_bin], axis=1)
    df_work = df_work.drop(columns=[col])

#%%

# Обычные категориальные признаки
cat_cols = [col for col in categorical_cols if col in df_work.columns]

for col in cat_cols:
    dummies = pd.get_dummies(df_work[col], prefix=col)
    for cn in dummies.columns:
        feature_groups[cn] = col
    df_work = pd.concat([df_work, dummies], axis=1)
    df_work = df_work.drop(columns=[col])

# Булевы колонки в int
bool_cols = df_work.select_dtypes(include=['bool']).columns
df_work[bool_cols] = df_work[bool_cols].astype(int)

# Связываем остальные незатронутые фичи (например, длины текстов)
for col in df_work.columns:
    if col not in feature_groups and col != 'ctr':
        feature_groups[col] = col

#%%

# ==== ОБУЧЕНИЕ Random Forest ====
X = df_work.drop(columns=['ctr'])
y = df_work['ctr']

rf = RandomForestRegressor(
    n_estimators=100, 
    max_depth=5 if n_samples < 30 else None, 
    random_state=42
)
rf.fit(X, y)

#%%
# --- 4. Расчет и агрегация важности ---
importances = rf.feature_importances_
feature_imp_df = pd.DataFrame({'feature_col': X.columns, 'importance': importances})
feature_imp_df['group'] = feature_imp_df['feature_col'].map(feature_groups)

grouped_imp = feature_imp_df.groupby('group')['importance'].sum().reset_index()
grouped_imp['importance_pct'] = (grouped_imp['importance'] / grouped_imp['importance'].sum()) * 100
grouped_imp = grouped_imp.sort_values(by='importance_pct', ascending=False).reset_index(drop=True)

#%%
plot=True
top_n = 10

# --- 5. Построение графика ---
if plot:
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    top_df = grouped_imp.head(top_n).copy()
    
    # Понятные бизнесу названия категорий
    name_map = {
        'image_phrases': 'Фразы на картинке',
        'image_text_length': 'Длина текста на баннере',
        'image_verbs': 'Глаголы на картинке',
        'image_nouns': 'Существительные на картинке',
        'image_tags': 'Теги / Детали изображения',
        'image_proportions': 'Пропорции картинки',
        'image_adjs': 'Прилагательные на картинке',
        'headline_nouns': 'Существительные в заголовке',
        'image_color_main': 'Главный цвет',
        'adtext_nouns': 'Существительные в тексте',
        'image_objects': 'Объекты на картинке',
        'image_color_rest': 'Вспомогательные цвета',
        'image_people_amount': 'Кол-во людей на картинке',
        'image_text_area': 'Объем текста на картинке'
    }
    top_df['readable_name'] = top_df['group'].map(lambda x: name_map.get(x, x))
    
    ax = sns.barplot(x='importance_pct', 
                     y='readable_name', 
                     data=top_df, 
                     palette='Blues_r',
                     hue='readable_name')
    plt.title(f'Степень влияния признаков на CTR (Размер выборки: {n_samples} баннеров)', fontsize=13, fontweight='bold')
    plt.xlabel('Степень влияния (%)')
    plt.ylabel('Признак / Категория')
    
    # Добавляем проценты на график
    for p in ax.patches:
        w = p.get_width()
        ax.annotate(f'{w:.1f}%', (w + 0.3, p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', fontsize=9, color='black')
                    
    plt.tight_layout()
    plt.show()

#%%
    

