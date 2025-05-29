import streamlit as st
import pandas as pd
import joblib
import numpy as np 

st.set_page_config(
    page_title="Предсказание цен на квартиры",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Загрузка обученного модели ---
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Файл модели не найден по пути: {model_path}. Убедитесь, что файл существует.")
        return None
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        return None

MODEL_PATH = 'my_apartment_xgb_model.joblib' 
model = load_model(MODEL_PATH)

# --- 2. Заголовок и описание приложения ---
st.title("Предсказание стоимости аренды квартиры 🏠")
st.write("""
Это приложение использует модель машинного обучения для предсказания
стоимости аренды квартиры в Нижнем Новгороде. Введите основные характеристики в меню слева.
Для некоторых параметров можно использовать медианные значения по умолчанию.
""")

# --- 3. Определение признаков и их медиан ---
all_expected_features = [
    'number_of_rooms', 'square', 'kitchen_square', 'extra', 'pets_allowed', 'floor',
    'floors_in_house', 'distance_to_center', 'cafes_count', 'bus_dist', 'sports_count',
    'school_dist', 'shops_count', 'culture_count', 'washing_machine', 'conditioner',
    'water_heater', 'balcony', 'balcony_loggia', 'loggia', 'payment_by_tags_included',
    'repair_designer', 'repair_euro', 'repair_needed', 'bathroom_separate',
    'batroom_combined_separate', 'room_isolated', 'room_isolated_connecting',
    'room_connecting', 'min_metro_time_5', 'min_metro_time_6', 'min_metro_time_11',
    'min_metro_time_16', 'min_metro_time_21', 'min_metro_time_31'
]

all_medians = {
    'number_of_rooms': 1.0, 'square': 43.0, 'kitchen_square': 9.0, 'extra': 0.0,
    'pets_allowed': 0.0, 'floor': 5.0, 'floors_in_house': 10.0,
    'distance_to_center': 8.053700, 'cafes_count': 11.0, 'bus_dist': 0.430000,
    'sports_count': 39.0, 'school_dist': 0.632500, 'shops_count': 26.0,
    'culture_count': 0.0, 'washing_machine': 0.0, 'conditioner': 0.0,
    'water_heater': 0.0, 'balcony': 0.0, 'balcony_loggia': 0.0, 'loggia': 0.0,
    'payment_by_tags_included': 0.0, 'repair_designer': 0.0, 'repair_euro': 0.0,
    'repair_needed': 0.0, 'bathroom_separate': 0.0, 'batroom_combined_separate': 0.0,
    'room_isolated': 0.0, 'room_isolated_connecting': 0.0, 'room_connecting': 0.0,
    'min_metro_time_5': 0.0, 'min_metro_time_6': 0.0, 'min_metro_time_11': 0.0,
    'min_metro_time_16': 0.0, 'min_metro_time_21': 0.0, 'min_metro_time_31': 0.0
}

for key, value in all_medians.items():
    is_binary_or_integer_count = key in [
        'washing_machine', 'conditioner', 'repair_designer', 'extra', 'pets_allowed',
        'water_heater', 'balcony', 'balcony_loggia', 'loggia', 'payment_by_tags_included',
        'repair_euro', 'repair_needed', 'bathroom_separate', 'batroom_combined_separate',
        'room_isolated', 'room_isolated_connecting', 'room_connecting',
        'min_metro_time_5', 'min_metro_time_6', 'min_metro_time_11',
        'min_metro_time_16', 'min_metro_time_21', 'min_metro_time_31'
    ] or (value == int(value) and key not in [
        'distance_to_center', 'bus_dist', 'school_dist', 'kitchen_square', 'square'
    ]) or key in ['cafes_count', 'culture_count', 'shops_count', 'sports_count', 'number_of_rooms', 'floor', 'floors_in_house']

    if is_binary_or_integer_count:
        all_medians[key] = int(value)

# Признаки, которые пользователь ВСЕГДА вводит напрямую
user_selectable_features_direct_input = [
    'square', 'washing_machine', 'distance_to_center', 'repair_designer',
    'conditioner', 'repair_euro', 'floors_in_house'
]

# Признаки, для которых пользователь МОЖЕТ ввести значение или использовать медиану
user_selectable_features_optional_input = [
    'cafes_count', 'culture_count'
]

# Скрытые признаки (Их значения по умолчанию будут медианами)
default_hidden_features_median = {
    k: v for k, v in all_medians.items()
    if k not in user_selectable_features_direct_input
    and k not in user_selectable_features_optional_input
    and k not in ['kitchen_square', 'number_of_rooms'] 
}


if model:
    st.sidebar.header("Укажите характеристики квартиры:")
    binary_options_map = {"Да": 1, "Нет": 0}
    user_inputs = {} 

    # --- Поля для признаков с прямым вводом (кроме kitchen_square и number_of_rooms) ---
    user_inputs['square'] = st.sidebar.number_input(
        "Общая площадь (кв. м)",
        min_value=10.0, max_value=500.0, value=float(all_medians['square']), step=0.5,
        key='square_input_main'
    )
    user_inputs['floors_in_house'] = st.sidebar.number_input(
        "Количество этажей в доме",
        min_value=1, max_value=100, value=int(all_medians['floors_in_house']), step=1,
        key='floors_in_house_input_main'
    )
    user_inputs['distance_to_center'] = st.sidebar.number_input(
        "Расстояние до центра города (км)",
        min_value=0.0, max_value=100.0, value=float(all_medians['distance_to_center']), step=0.1,
        key='distance_to_center_input_main'
    )
    st.sidebar.markdown("---")
    washing_machine_label = st.sidebar.radio(
        "Есть посудомоечная машина?",
        options=["Нет", "Да"], index=int(all_medians['washing_machine']), key='washing_machine_label_main'
    )
    user_inputs['washing_machine'] = binary_options_map[washing_machine_label]

    repair_designer_label = st.sidebar.radio(
        "Ремонт дизайнерский?",
        options=["Нет", "Да"], index=int(all_medians['repair_designer']), key='repair_designer_label_main'
    )
    user_inputs['repair_designer'] = binary_options_map[repair_designer_label]

    conditioner_label = st.sidebar.radio(
        "Есть кондиционер?",
        options=["Нет", "Да"], index=int(all_medians['conditioner']), key='conditioner_label_main'
    )
    user_inputs['conditioner'] = binary_options_map[conditioner_label]

    repair_euro_label = st.sidebar.radio(
        "Ремонт типа 'евро'?",
        options=["Нет", "Да"], index=int(all_medians['repair_euro']), key='repair_euro_label_main'
    )
    user_inputs['repair_euro'] = binary_options_map[repair_euro_label]
    st.sidebar.markdown("---")

    # --- Логика для "Студии" ---
    is_studio = st.sidebar.checkbox(
        "Это студия?",
        key='is_studio_cb',
        value=False, 
        help="Если отмечено, количество комнат будет 1, а площадь кухни для модели будет 0."
    )

    if is_studio:
        user_inputs['number_of_rooms'] = 1
        user_inputs['kitchen_square'] = 0.0
        st.sidebar.markdown(f"ㅤ➥ *Количество комнат (студия): **1***")
        st.sidebar.markdown(f"ㅤ➥ *Площадь кухни (студия): **0.0 кв. м***")
    else:
        user_inputs['number_of_rooms'] = st.sidebar.number_input(
            "Количество комнат (1-комн. квартира = 1)",
            min_value=1, max_value=10, value=int(all_medians.get('number_of_rooms', 1)), step=1,
            key='number_of_rooms_input_main_regular'
        )
        user_inputs['kitchen_square'] = st.sidebar.number_input(
            "Площадь кухни (кв. м)",
            min_value=0.0, max_value=100.0, value=float(all_medians.get('kitchen_square', 9.0)), step=0.5,
            key='kitchen_square_input_main_regular'
        )
    st.sidebar.markdown("---")

    # --- Поля с опциональным вводом для cafes_count и culture_count ---
    for feature_key in user_selectable_features_optional_input:
        median_value = all_medians[feature_key]
        feature_display_name = feature_key.replace('_', ' ').capitalize()

        if feature_key == 'cafes_count':
            label_text = "Кол-во кафе/ресторанов в радиусе 1 км"
            max_val = 300 
        elif feature_key == 'culture_count':
            label_text = "Кол-во объектов культуры в радиусе 1 км"
            max_val = 200 
        else:
            label_text = feature_display_name
            max_val = 200

        use_custom_value = st.sidebar.checkbox(
            f"Указать свое значение для '{label_text}' (медиана: {median_value})",
            key=f'use_custom_{feature_key}_cb',
            value=False 
        )
        if use_custom_value:
            user_inputs[feature_key] = st.sidebar.number_input(
                label_text,
                min_value=0, max_value=max_val, value=int(median_value), step=1,
                key=f'{feature_key}_input_custom'
            )
        else:
            user_inputs[feature_key] = int(median_value) 
        st.sidebar.markdown("---")


    # --- 4. Кнопка для предсказания ---
    if st.sidebar.button("✨ Предсказать стоимость ✨"):
        # --- 5. Подготовка ПОЛНОГО набора данных для модели ---
        final_input_data = default_hidden_features_median.copy()
        final_input_data.update(user_inputs) 

        input_df_ordered_values = [final_input_data.get(name, pd.NA) for name in all_expected_features]
        input_df = pd.DataFrame([input_df_ordered_values], columns=all_expected_features)

        # --- 6. Получение предсказания ---
        try:
            prediction = model.predict(input_df)
            predicted_price = prediction[0]

            formatted_predicted_price_string = f"{predicted_price:,.0f}".replace(',', ' ')

            st.subheader("Результат предсказания:")
            st.metric(label="Ориентировочная стоимость аренды", value=f"{formatted_predicted_price_string} ₽/месяц")
            st.success(f"Модель предсказывает стоимость аренды примерно в **{formatted_predicted_price_string} ₽** в месяц (с учетом параметров по умолчанию).")

            if is_studio:
                st.info("Предсказание сделано с учетом того, что это студия (1 комната, площадь кухни 0 кв.м).")

        except Exception as e:
            st.error(f"Произошла ошибка при получении предсказания: {e}")
            st.error("Убедитесь, что все поля заполнены корректно и модель обучена на признаках с такими же именами и типами данных.")
            st.error(f"Имена колонок, переданные в модель: {input_df.columns.tolist()}")
else:
    if not model and MODEL_PATH:
         st.error("🚫 Модель не загружена. Проверьте сообщения об ошибках выше.")
         st.info(f"Ожидался файл модели по пути: {MODEL_PATH}")
