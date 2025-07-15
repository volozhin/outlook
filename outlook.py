import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import csv
from collections import Counter
from datetime import datetime

# ========== Логика анализа и вспомогательные функции ==========
def detect_delimiter(uploaded_file, num_lines=5):
    uploaded_file.seek(0)
    sample = ''.join([uploaded_file.readline().decode('utf-8', errors='ignore') for _ in range(num_lines)])
    uploaded_file.seek(0)  # вернуть указатель файла в начало
    dialect = csv.Sniffer().sniff(sample)
    return dialect.delimiter

def preprocess(df):
    df = df.copy()
    # Определение предполагаемого сотрудника по частоте 'От кого'
    senders = df['От кого'].dropna().value_counts()
    probable_sender = senders.index[0] if not senders.empty else ""
    employee_name = probable_sender.split("<")[0].strip()
    # Приведение столбцов с датами к типу datetime
    df['Дата получения'] = pd.to_datetime(df['Дата получения'], errors='coerce')
    df['Дата отправки'] = pd.to_datetime(df['Дата отправки'], errors='coerce')
    # Очистка темы письма от префиксов "Re:"/"FW:"
    df['Тема чистая'] = df['Тема'].str.lower().str.replace(r"^re:\s*|fw:\s*", "", regex=True).str.strip()
    # Определение роли сотрудника в письме
    df['Роль'] = df.apply(lambda row: (
        'Отправитель' if employee_name != "" and employee_name in str(row['От кого']) else
        'Получатель' if employee_name != "" and employee_name in str(row['Кому']) else
        'В копии'   if employee_name != "" and employee_name in str(row['Копия']) else
        'Не указана'), axis=1)
    # Выделение года, номера недели и часа получения письма
    df['Год'] = df['Дата получения'].dt.isocalendar().year
    df['Неделя'] = df['Дата получения'].dt.isocalendar().week
    df['Час'] = df['Дата получения'].dt.hour
    return df

def reply_time_analysis(df):
    import numpy as np
    # Разделение на входящие (где анализируемый сотрудник – получатель) и исходящие
    incoming = df[df['Роль'] == 'Получатель'].copy()
    outgoing = df[df['Роль'] == 'Отправитель'].copy()
    # Убираем информацию о временной зоне для корректного расчета разницы
    incoming['Дата получения'] = incoming['Дата получения'].dt.tz_localize(None)
    outgoing['Дата отправки'] = outgoing['Дата отправки'].dt.tz_localize(None)
    results = []
    for _, row in incoming.iterrows():
        # Для каждого входящего письма ищем исходящее с тем же ConversationID, отправленное позже
        out_same_conv = outgoing[outgoing['ConversationID'] == row['ConversationID']]
        after = out_same_conv[out_same_conv['Дата отправки'] > row['Дата получения']]
        if not after.empty:
            soonest = after.sort_values('Дата отправки').iloc[0]
            delta_hours = (soonest['Дата отправки'] - row['Дата получения']).total_seconds() / 3600
            # Рассчитываем количество рабочих дней между датами (np.busday_count учитывает будни)
            start = row['Дата получения'].date()
            end = soonest['Дата отправки'].date()
            business_days = np.busday_count(start, end)
            results.append({
                'ConversationID': row['ConversationID'],
                'Неделя': row['Неделя'],
                'Тема': row['Тема'],
                'Дата входящего': row['Дата получения'],
                'Дата ответа': soonest['Дата отправки'],
                'Время ответа': delta_hours,
                'Время ответа (дней)': round(delta_hours / 24, 2),
                'Рабочих дней до ответа': business_days,
                'Просрочка': business_days > 3
            })
    return pd.DataFrame(results)

def plot_reply_distribution(reply_times):
    # Распределение времени ответа (отсекаем значения больше 7 дней/168 часов)
    clean_times = [t for t in reply_times if t <= 168]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(clean_times, bins=30, edgecolor='black')
    ax.set_xticks(range(0, 169, 24))
    ax.set_title("Распределение времени ответа (до 7 дней)")
    ax.set_xlabel("Время ответа, часы")
    ax.set_ylabel("Количество писем")
    return fig

def plot_weekly_trends(reply_df):
    # Среднее и медианное время ответа по неделям
    summary = reply_df.groupby('Неделя')['Время ответа'].agg(['mean', 'median'])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(summary.index, summary['mean'], marker='o', label='Среднее время')
    ax.plot(summary.index, summary['median'], marker='s', label='Медианное время')
    ax.set_title("Среднее и медианное время ответа по неделям")
    ax.set_xlabel("Неделя")
    ax.set_ylabel("Время ответа, часы")
    ax.legend()
    ax.grid(True)
    return fig

def plot_weekly_message_flow(df):
    # Еженедельный поток писем: получено vs отправлено
    weekly = df[df['Роль'].isin(['Отправитель', 'Получатель'])]
    pivot = weekly.pivot_table(index='Неделя', columns='Роль', aggfunc='size', fill_value=0)
    # Процент ответов (исходящие относительно входящих) по неделям
    pivot['Процент ответов'] = (pivot['Отправитель'] / pivot['Получатель'] * 100).round(1)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(pivot.index)), pivot['Получатель'], width=0.4, label='Получено')
    ax.bar([i + 0.4 for i in range(len(pivot.index))], pivot['Отправитель'], width=0.4, label='Отправлено')
    # Подписи процентов над столбиками
    for i, week in enumerate(pivot.index):
        ax.text(i, max(pivot.iloc[i]['Получатель'], pivot.iloc[i]['Отправитель']) + 1,
                f"{pivot.iloc[i]['Процент ответов']}%", ha='center', fontsize=9)
    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.index)
    ax.set_title("Объем входящих и исходящих писем по неделям")
    ax.set_xlabel("Неделя")
    ax.set_ylabel("Количество писем")
    ax.legend()
    ax.grid(axis='y')
    return fig

def plot_weekly_claims(df):
    # Анализ писем с темой "рекламация" по неделям
    df['Рекламация'] = df['Тема'].str.lower().str.contains("рекламация")
    weekly_claims = df.groupby(['Неделя', 'Рекламация']).size().unstack(fill_value=0)
    # Переименуем столбцы для ясности
    if True in weekly_claims.columns:
        weekly_claims.columns = ['Без рекламации', 'С рекламацией']
    else:
        weekly_claims.columns = ['Без рекламации']
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.4
    weeks = weekly_claims.index
    x = range(len(weeks))
    # Столбики для писем без рекламаций и с рекламациями
    ax.bar([i - bar_width/2 for i in x], weekly_claims['Без рекламации'], width=bar_width, label='Без рекламации')
    if 'С рекламацией' in weekly_claims.columns:
        ax.bar([i + bar_width/2 for i in x], weekly_claims['С рекламацией'], width=bar_width, label='С рекламацией')
    ax.set_xticks(x)
    ax.set_xticklabels(weeks)
    ax.set_title("Количество писем с темой 'рекламация' по неделям")
    ax.set_xlabel("Неделя")
    ax.set_ylabel("Количество писем")
    ax.legend()
    ax.grid(axis='y')
    return fig

def stretch_topics(df):
    # Выявление "растянутых" тем переписки (длительностью 20+ дней) по ConversationID
    stretched_list = []
    # Рассматриваем только переписки, где сотрудник участвовал как отправитель
    conv_with_outgoing = set(df[df['Роль'] == 'Отправитель']['ConversationID'].unique())
    for conv_id, group in df.groupby('ConversationID'):
        if conv_id not in conv_with_outgoing:
            continue
        # Убираем tz-информацию из дат для корректного сравнения
        try:
            group['Дата отправки'] = group['Дата отправки'].dt.tz_localize(None)
        except:
            pass
        try:
            group['Дата получения'] = group['Дата получения'].dt.tz_localize(None)
        except:
            pass
        # Находим самое раннее и самое позднее время в переписке
        min_received = group['Дата получения'].min()
        min_sent = group['Дата отправки'].min()
        max_received = group['Дата получения'].max()
        max_sent = group['Дата отправки'].max()
        if pd.isna(min_received):
            earliest = min_sent
        elif pd.isna(min_sent):
            earliest = min_received
        else:
            earliest = min(min_received, min_sent)
        if pd.isna(max_received):
            latest = max_sent
        elif pd.isna(max_sent):
            latest = max_received
        else:
            latest = max(max_received, max_sent)
        duration = (latest - earliest) if pd.notna(earliest) and pd.notna(latest) else pd.Timedelta(0)
        if duration >= pd.Timedelta(days=20):
            count_messages = len(group)
            # Представительная тема переписки: берем тему первого сообщения по времени
            rep_subject = group.sort_values(
                ['Дата получения', 'Дата отправки']).iloc[0]['Тема чистая']
            stretched_list.append({
                'ConversationID': conv_id,
                'Тема чистая': rep_subject,
                'Длительность': duration,
                'Количество писем': count_messages
            })
    stretched_df = pd.DataFrame(stretched_list)
    return stretched_df.sort_values(by='Длительность', ascending=False).reset_index(drop=True)

def normalize_name(name):
    # Нормализация имени сотрудника: убрать лишние пробелы, привести к нижнему регистру
    name = ' '.join(str(name).strip().split())
    return name.lower()

def plot_department_traffic(incoming_summary, outgoing_summary):
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    incoming_summary.sort_values().plot(kind='barh', ax=ax1, color='#1f77b4')
    ax1.set_title('Входящие письма', fontsize=10)
    ax1.set_xlabel('Писем', fontsize=8)
    ax1.set_ylabel('Отдел', fontsize=8)
    ax1.tick_params(axis='x', labelsize=7)
    ax1.tick_params(axis='y', labelsize=7)
    ax1.grid(axis='x')

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    outgoing_summary.sort_values().plot(kind='barh', ax=ax2, color='#ff7f0e')
    ax2.set_title('Исходящие письма', fontsize=10)
    ax2.set_xlabel('Писем', fontsize=8)
    ax2.set_ylabel('Отдел', fontsize=8)
    ax2.tick_params(axis='x', labelsize=7)
    ax2.tick_params(axis='y', labelsize=7)
    ax2.grid(axis='x')

    return fig1, fig2

# ========== Streamlit UI ==========

st.set_page_config(layout="wide")
st.title("Дашборд анализа почты сотрудника")

# 1. Загрузка и обработка данных писем
uploaded_file = st.file_uploader("Загрузите CSV-файл с выгрузкой почты", type="csv")
if uploaded_file:
    # Определяем разделитель и считываем CSV с письмами
    delimiter = detect_delimiter(uploaded_file)
    df = pd.read_csv(uploaded_file, delimiter=delimiter, encoding='utf-8')
    df = preprocess(df)

    # Фильтрация по году и неделе (опционально)
    years = sorted(df['Год'].dropna().unique().tolist())
    weeks = sorted(df['Неделя'].dropna().unique().tolist())
    selected_year = st.selectbox("Выберите год", options=['Все'] + years)
    selected_week = st.selectbox("Выберите неделю (или 'Все')", options=['Все'] + weeks)
    if selected_year != 'Все':
        df = df[df['Год'] == selected_year]
    if selected_week != 'Все':
        df = df[df['Неделя'] == selected_week]

    # 2. Основные метрики и графики по письмам
    st.subheader("📊 Распределение писем по ролям")
    role_counts = df['Роль'].value_counts().reset_index().rename(columns={'index': 'Роль', 'Роль': 'Количество'})
    st.dataframe(role_counts)

    st.subheader("⏱️ Время ответа")
    reply_df = reply_time_analysis(df)
    if not reply_df.empty:
        st.metric("Среднее время ответа (ч)", round(reply_df['Время ответа'].mean(), 2))
        st.metric("Медианное время ответа (ч)", round(reply_df['Время ответа'].median(), 2))

        # 🔹 Считаем количество писем по каждой переписке (ConversationID)
        message_counts = []
        for conv_id in reply_df['ConversationID'].unique():
            count = df[df['ConversationID'] == conv_id].shape[0]
            message_counts.append(count)

        if message_counts:
            st.metric("Среднее количество писем на инцидент", round(pd.Series(message_counts).mean(), 1))
            st.subheader("📈 Распределение количества писем на инцидент")
            fig_msg, ax = plt.subplots(figsize=(8, 2))
            ax.hist(message_counts, bins=range(1, max(message_counts) + 2), edgecolor='black', align='left')
            ax.set_title("Распределение количества писем на инцидент", fontsize=10)
            ax.set_xlabel("Количество писем", fontsize=8)
            ax.set_ylabel("Частота", fontsize=8)
            ax.tick_params(axis='x', labelsize=7)
            ax.tick_params(axis='y', labelsize=7)
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            st.pyplot(fig_msg)

        st.metric("Мин. время ответа (ч)", round(reply_df['Время ответа'].min(), 2))
        st.metric("Макс. время ответа (ч)", round(reply_df['Время ответа'].max(), 2))

        message_counts = []
        for conv_id in reply_df['ConversationID'].unique():
            count = df[df['ConversationID'] == conv_id].shape[0]
            message_counts.append(count)

        if message_counts:
            st.metric("Среднее количество писем на инцидент", round(pd.Series(message_counts).mean(), 1))
            st.metric("Медианное количество писем на инцидент", round(pd.Series(message_counts).median(), 1))

        st.subheader("📬 Аналитические графики по времени ответа")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_reply_distribution(reply_df['Время ответа']))
        with col2:
            st.pyplot(plot_weekly_trends(reply_df))
    else:
        st.info("Нет данных для анализа времени ответа.")

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_weekly_message_flow(df))
    with col2:
        st.pyplot(plot_weekly_claims(df))

    st.subheader("📌 Растянутые темы (переписка длилась ≥20 дней)")
    stretched_df = stretch_topics(df)
    st.dataframe(stretched_df.head(10))

    selected_topic = st.selectbox("Выберите тему для детализации переписки", options=stretched_df['Тема чистая'] if not stretched_df.empty else [])
    if selected_topic:
        st.subheader("📥 Детализация переписки по выбранной теме")
        # Находим соответствующий ConversationID для выбранной темы
        conv_ids = stretched_df[stretched_df['Тема чистая'] == selected_topic]['ConversationID'].tolist()
        if conv_ids:
            conv_id = conv_ids[0]
            topic_messages = df[df['ConversationID'] == conv_id][[
                'Дата получения', 'Дата отправки', 'От кого', 'Кому', 'Копия', 'Тема', 'Роль'
            ]]
            st.dataframe(topic_messages.sort_values(by='Дата отправки'))
        else:
            st.info("Переписка не найдена.")

    st.subheader("⚠ Письма с ответом позже 3 рабочих дней")
    late_replies = reply_df[reply_df['Просрочка'] == True]
    if not late_replies.empty:
        st.dataframe(late_replies[['Тема', 'Дата входящего', 'Дата ответа', 'Время ответа', 'Время ответа (дней)']])
    else:
        st.success("Все входящие письма были обработаны вовремя.")

    # 3. Анализ взаимодействий по отделам (отдельный блок)
    st.subheader("📨 Анализ взаимодействий по отделам")
    employees_file = st.file_uploader("Загрузите CSV-файл со списком сотрудников", type="csv")
    if employees_file:
        # Считываем файл сотрудников с авто-определением разделителя
        delim_emp = detect_delimiter(employees_file)
        employees_df = pd.read_csv(employees_file, delimiter=delim_emp, encoding='utf-8')
        # Нормализуем ФИО сотрудников для сопоставления
        employees_df['ФИО_норм'] = employees_df['ФИО'].apply(normalize_name)

        # Сбор всех имен (входящая почта: От кого, Кому, Копия)
        all_names = []
        for col in ['От кого', 'Кому', 'Копия']:
            df[col] = df[col].fillna("")
            for entry in df[col]:
                parts = str(entry).split(';')
                for name in parts:
                    norm_name = normalize_name(name)
                    if norm_name and 'нет данных' not in norm_name and '@' not in norm_name:
                        all_names.append(norm_name)
        name_counts = Counter(all_names)
        name_df = pd.DataFrame(name_counts.items(), columns=['ФИО_норм', 'Количество писем'])

        # Объединяем с подразделениями и фильтруем
        merged_df = name_df.merge(employees_df[['ФИО_норм', 'Подразделение']], on='ФИО_норм', how='left')
        filtered_df = merged_df.dropna(subset=['Подразделение'])
        filtered_df_no_claims = filtered_df[filtered_df['Подразделение'].str.lower() != 'претензионный отдел']

        # Группируем по отделам – входящие письма (для топ-10 отделов)
        department_summary_no_claims = (
            filtered_df_no_claims.groupby('Подразделение')['Количество писем']
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        # Сбор всех имен адресатов (исходящая почта: Кому и Копия)
        outgoing_names = []
        for col in ['Кому', 'Копия']:
            for entry in df[col]:
                parts = str(entry).split(';')
                for name in parts:
                    norm_name = normalize_name(name)
                    if norm_name and 'нет данных' not in norm_name and '@' not in norm_name:
                        outgoing_names.append(norm_name)
        outgoing_counts = Counter(outgoing_names)
        outgoing_df = pd.DataFrame(outgoing_counts.items(), columns=['ФИО_норм', 'Количество писем'])
        outgoing_merged = outgoing_df.merge(employees_df[['ФИО_норм', 'Подразделение']], on='ФИО_норм', how='left')
        outgoing_filtered = outgoing_merged.dropna(subset=['Подразделение'])
        outgoing_filtered = outgoing_filtered[outgoing_filtered['Подразделение'].str.lower() != 'претензионный отдел']

        outgoing_summary = (
            outgoing_filtered.groupby('Подразделение')['Количество писем']
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        # Отрисовка графиков топ-10 отделов для входящей и исходящей почты
        fig1, fig2 = plot_department_traffic(department_summary_no_claims, outgoing_summary)
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig1)
        with col2:
            st.pyplot(fig2)
    else:
        st.info("👉 Загрузите файл со списком сотрудников для анализа коммуникаций по отделам.")
