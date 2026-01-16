import time
import json
import logging
import os
import requests
import cloudscraper
from datetime import datetime
from typing import List, Set, Dict
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import random
import pickle

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Настройки Telegram
BOT_TOKEN = "7877159131:AAGrC_QlzSvKO1n_AFkJlMY7-UXTx_1l590"
CHAT_ID = "217141303"
DATA_FILE = "predator_ai_data.json"
MODEL_FILE = "model_rfc.pkl"
CHECK_INTERVAL_SECONDS = 180
REQUEST_DELAY_MIN = 3
REQUEST_DELAY_MAX = 6
RETRY_ATTEMPTS = 3
RETRY_ATTEMPTS_QUICK = 1
CACHE_TIMEOUT = 300
BASE_URL = "https://api.sofascore.com/api/v1/"
MAX_REQUEST_DELAY = 60

BLACKLIST_KEYWORDS = [
    "u19", "u20", "u21", "reserve", "reserves", "youth", "academy", "under-",
    "women", "womens", "female", "ladies",
    "friendly", "exhibition", "test match", "preseason", "practice",
    "cup", "copa", "pokale", "coppa", "pokal", "fa cup", "coupe", "dfb-pokal"
]

clf = None
CACHE = {}

total_signals_ever = 0
successful_signals_ever = 0
new_signals_in_cycle = 0


def extract_match_minute(event_data: dict) -> int:
    time_info = event_data.get("time", {})
    for key in ["minute", "currentMatchMinute"]:
        val = time_info.get(key)
        if val is not None:
            try:
                minute = int(val)
                if 0 <= minute <= 120:
                    return minute
            except (ValueError, TypeError):
                pass
    start_ts = time_info.get("currentPeriodStartTimestamp")
    if start_ts:
        try:
            minute = int((time.time() - start_ts) // 60)
            if 0 <= minute <= 120:
                return minute
        except Exception:
            pass
    return 0


def save_local_data(pending_targets: List[dict], sent_notifications: Set[int], attack_data_samples: List[list], attack_labels: List[int]):
    data = {
        "pending_targets": pending_targets,
        "sent_notifications": list(sent_notifications),
        "attack_data_samples": attack_data_samples,
        "attack_labels": attack_labels,
    }
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Локальные данные сохранены.")
        print("💾 Локальные данные сохранены в файл.")
    except Exception as e:
        logger.error(f"Ошибка сохранения данных: {str(e)}")
        print(f"❌ Ошибка сохранения данных: {str(e)}")


def load_local_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
                return (
                    data.get("pending_targets", []),
                    set(data.get("sent_notifications", [])),
                    data.get("attack_data_samples", []),
                    data.get("attack_labels", []),
                )
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {str(e)}")
            print(f"❌ Ошибка загрузки данных: {str(e)}")
    return [], set(), [], []


def load_model():
    global clf
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "rb") as f:
                clf = pickle.load(f)
            logger.info("Загружена существующая модель RandomForest.")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели из файла {MODEL_FILE}: {str(e)}")
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        logger.info("Создана новая модель RandomForest.")


def get_from_cache(key: str) -> Dict:
    if key in CACHE:
        entry = CACHE[key]
        if time.time() - entry["timestamp"] < CACHE_TIMEOUT:
            return entry["data"]
        else:
            del CACHE[key]
    return None


def set_to_cache(key: str, data: Dict):
    CACHE[key] = {"data": data, "timestamp": time.time()}


def make_request_with_retry(scraper, url: str, max_attempts: int = RETRY_ATTEMPTS):
    current_delay = random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)
    for attempt in range(max_attempts):
        try:
            response = scraper.get(url)
            response.raise_for_status()
            time.sleep(current_delay)
            return response
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [403, 404]:
                logger.warning(f"Ошибка {e.response.status_code} на попытке {attempt+1}/{max_attempts}, пропуск.")
                if max_attempts == RETRY_ATTEMPTS_QUICK:
                    break
            else:
                logger.error(f"HTTP ошибка {attempt+1}/{max_attempts}: {str(e)}")
            if attempt < max_attempts -1:
                time.sleep(current_delay)
                current_delay = min(current_delay * 2, MAX_REQUEST_DELAY)
        except Exception as e:
            logger.error(f"Ошибка {attempt+1}/{max_attempts}: {str(e)}")
            if attempt < max_attempts -1:
                time.sleep(current_delay)
                current_delay = min(current_delay * 2, MAX_REQUEST_DELAY)
    logger.error(f"Не удалось выполнить запрос после {max_attempts} попыток.")
    return None


def get_live_match_events(scraper: cloudscraper.CloudScraper) -> List[Dict]:
    cache_key = "live_events"
    cached_data = get_from_cache(cache_key)
    if cached_data is not None:
        logger.info("Использую кэш live-матчей.")
        return cached_data
    url = f"{BASE_URL}sport/football/events/live"
    response = make_request_with_retry(scraper, url)
    if response:
        try:
            data = response.json().get("events", [])
            set_to_cache(cache_key, data)
            logger.info("Получены live-матчи.")
            return data
        except Exception as e:
            logger.error(f"Ошибка разбора live-матчей: {str(e)}")
    return []


def get_full_event_data(scraper: cloudscraper.CloudScraper, match_id: int) -> Dict:
    cache_key = f"event_{match_id}"
    cached_data = get_from_cache(cache_key)
    if cached_data is not None:
        logger.info(f"Использую кэш матча {match_id}.")
        return cached_data
    url = f"{BASE_URL}event/{match_id}"
    response = make_request_with_retry(scraper, url)
    if response:
        try:
            data = response.json().get("event", {})
            set_to_cache(cache_key, data)
            logger.info(f"Получены данные матча {match_id}.")
            return data
        except Exception as e:
            logger.error(f"Ошибка разбора матча {match_id}: {str(e)}")
    return {}


def get_match_statistics(scraper: cloudscraper.CloudScraper, match_id: int) -> Dict:
    cache_key = f"stats_{match_id}"
    cached_data = get_from_cache(cache_key)
    if cached_data is not None:
        logger.info(f"Использую кэш статистики матча {match_id}.")
        return cached_data
    url = f"{BASE_URL}event/{match_id}/statistics"
    response = make_request_with_retry(scraper, url, max_attempts=RETRY_ATTEMPTS_QUICK)
    if response:
        try:
            stats = response.json().get("statistics", [])
            data = stats[0] if stats else {}
            set_to_cache(cache_key, data)
            logger.info(f"Получены данные статистики матча {match_id}.")
            return data
        except Exception as e:
            logger.error(f"Ошибка статистики матча {match_id}: {str(e)}")
    return {}


def get_match_incidents(scraper: cloudscraper.CloudScraper, match_id: int) -> List[Dict]:
    cache_key = f"incidents_{match_id}"
    cached_data = get_from_cache(cache_key)
    if cached_data is not None:
        logger.info(f"Использую кэш событий матча {match_id}.")
        return cached_data
    url = f"{BASE_URL}event/{match_id}/incidents"
    response = make_request_with_retry(scraper, url)
    if response:
        try:
            data = response.json().get("incidents", [])
            set_to_cache(cache_key, data)
            logger.info(f"Получены события матча {match_id}.")
            return data
        except Exception as e:
            logger.error(f"Ошибка событий матча {match_id}: {str(e)}")
    return []


def extract_features(stats_data: dict, event_data: dict) -> List[float]:
    # Можно расширять признаки, сейчас базовые
    shots_on_target = 0.0
    corners = 0.0
    possession_diff = 0.0
    for group in stats_data.get("groups", []):
        for item in group.get("statisticsItems", []):
            name = item.get("name", "")
            try:
                if name == "Shots on target":
                    shots_on_target = (
                        float(item.get("home", "0").replace("%", "0"))
                        + float(item.get("away", "0").replace("%", "0"))
                    )
                elif name == "Corner kicks":
                    corners = (
                        float(item.get("home", "0").replace("%", "0"))
                        + float(item.get("away", "0").replace("%", "0"))
                    )
                elif name == "Ball possession":
                    p_home = float(item.get("home", "0%").replace("%", "0"))
                    p_away = float(item.get("away", "0%").replace("%", "0"))
                    possession_diff = abs(p_home - p_away)
            except Exception:
                pass

    current_minute = extract_match_minute(event_data)

    return [shots_on_target, corners, possession_diff, current_minute]


def format_statistics(stats_data: dict, incidents_data: List[Dict]) -> str:
    sot, corners, possession, shots_total, shots_off_target, offsides, free_kicks = (
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    )
    for group in stats_data.get("groups", []):
        for item in group.get("statisticsItems", []):
            if item.get("name") == "Shots on target":
                sot = (int(item.get("home", "0")), int(item.get("away", "0")))
            if item.get("name") == "Corner kicks":
                corners = (int(item.get("home", "0")), int(item.get("away", "0")))
            if item.get("name") == "Ball possession":
                possession = (
                    int(item.get("home", "0%").replace("%", "0")),
                    int(item.get("away", "0%").replace("%", "0")),
                )
            if item.get("name") == "Shots":
                shots_total = (int(item.get("home", "0")), int(item.get("away", "0")))
            if item.get("name") == "Shots off target":
                shots_off_target = (
                    int(item.get("home", "0")),
                    int(item.get("away", "0")),
                )
            if item.get("name") == "Offsides":
                offsides = (int(item.get("home", "0")), int(item.get("away", "0")))
            if item.get("name") == "Free kicks":
                free_kicks = (int(item.get("home", "0")), int(item.get("away", "0")))

    yellow_cards_home, yellow_cards_away, red_cards_home, red_cards_away = 0, 0, 0, 0
    for incident in incidents_data:
        if incident.get("type") == "card":
            if incident.get("color") == "yellow":
                if incident.get("isHome"):
                    yellow_cards_home += 1
                else:
                    yellow_cards_away += 1
            elif incident.get("color") == "red":
                if incident.get("isHome"):
                    red_cards_home += 1
                else:
                    red_cards_away += 1

    stats_text = f"<b>Статистика матча:</b>\n"
    stats_text += f"🏹 <b>Удары в створ:</b> {sot[0]} - {sot[1]}\n"
    stats_text += f"⚽ <b>Удары всего:</b> {shots_total[0]} - {shots_total[1]}\n"
    stats_text += f"❌ <b>Удары мимо:</b> {shots_off_target[0]} - {shots_off_target[1]}\n"
    stats_text += f"📐 <b>Угловые:</b> {corners[0]} - {corners[1]}\n"
    stats_text += f"⚖️ <b>Владение мячом:</b> {possession[0]}% - {possession[1]}%\n"
    stats_text += f"🚷 <b>Офсайды:</b> {offsides[0]} - {offsides[1]}\n"
    stats_text += f"🦶 <b>Штрафные:</b> {free_kicks[0]} - {free_kicks[1]}\n"
    stats_text += f"🟨 <b>Жёлтые карточки:</b> {yellow_cards_home} - {yellow_cards_away}\n"
    stats_text += f"🟥 <b>Красные карточки:</b> {red_cards_home} - {red_cards_away}"
    return stats_text


def train_model(samples: List[list], labels: List[int]):
    global clf
    if len(samples) > 10 and len(samples) == len(labels):
        try:
            expected_length = len(samples[0]) if samples else 4
            filtered_samples = []
            filtered_labels = []
            for i in range(len(samples)):
                if len(samples[i]) == expected_length:
                    filtered_samples.append(samples[i])
                    filtered_labels.append(labels[i])
            if len(filtered_samples) > 10:
                X = np.array(filtered_samples)
                y = np.array(filtered_labels)
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X, y)
                with open(MODEL_FILE, "wb") as f:
                    pickle.dump(clf, f)
                logger.info(f"Модель RandomForest обучена и сохранена. Использовано {len(filtered_samples)} примеров из {len(samples)}.")
            else:
                logger.info("Недостаточно данных для обучения после фильтрации.")
        except Exception as e:
            logger.error(f"Ошибка обучения модели: {str(e)}")
    else:
        logger.info("Недостаточно данных для обучения модели или несовпадение размеров данных и меток.")


def strategy_logistic_regression(features: List[float], event_data: dict) -> (bool, float):
    # Переименуем чтобы отражать RandomForest
    goals_first_half = event_data.get("homeScore", {}).get("period1", 0) + event_data.get("awayScore", {}).get("period1", 0)
    if goals_first_half > 0:
        return False, 0.0
    global clf
    if not clf or not hasattr(clf, "predict_proba"):
        prob = 0.5 if features[0] >= 1 else 0.0
        return prob > 0.1, prob
    try:
        prob = clf.predict_proba([features])[0][1]
        return prob > 0.1, prob
    except Exception:
        prob = 0.5 if features[0] >= 1 else 0.0
        return prob > 0.1, prob


def strategy_shots_corners(features: List[float]) -> (bool, float):
    shots_on_target = features[0]
    corners = features[1]
    triggered = shots_on_target > 7 or corners > 5
    prob = min(1.0, (shots_on_target * 0.1 + corners * 0.05))
    return triggered, prob


def strategy_possession_attack(features: List[float]) -> (bool, float):
    shots_on_target = features[0]
    possession_diff = features[2]
    triggered = possession_diff > 40.0 and shots_on_target > 3
    prob = 0.8 if triggered else 0.0
    return triggered, prob


def send_telegram(message: str, match_id: int = None, is_success_report: bool = False, is_status_update: bool = False) -> bool:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    params = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    if match_id:
        keyboard = {
            "inline_keyboard": [
                [{"text": "🔗 Открыть на Sofascore", "url": f"https://www.sofascore.com/event/{match_id}"}],
                [{"text": "📊 Открыть в 1xBet", "url": "https://1xbetkz.mobi/ru"}],
            ]
        }
        params["reply_markup"] = json.dumps(keyboard)
    current_delay = REQUEST_DELAY_MIN
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                if "Бот запущен" in message:
                    logger.info("Отправлено уведомление о запуске.")
                    print("🚀 Отправлено уведомление о запуске.")
                elif is_success_report:
                    logger.info(f"ЦЕЛЬ {match_id} ПОРАЖЕНА! Отправлен победный отчет.")
                    print(f"✅ ЦЕЛЬ {match_id} ПОРАЖЕНА! Отправлен победный отчет.")
                elif is_status_update:
                    logger.info(f"ОБНОВЛЕНИЕ СТАТУСА матча {match_id}! Отправлен отчет.")
                    print(f"⚠️ ОБНОВЛЕНИЕ СТАТУСА матча {match_id}! Отправлен отчет.")
                elif "Обнаружена цель" in message:
                    logger.info(f"ОБНАРУЖЕНА ЦЕЛЬ {match_id}! Отправлен сигнал.")
                    print(f"🔥 ОБНАРУЖЕНА ЦЕЛЬ {match_id}! Отправлен сигнал.")
                return True
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(current_delay)
                current_delay *= 2
        except Exception as e:
            logger.error(
                f"Ошибка отправки в Telegram (попытка {attempt + 1}/{RETRY_ATTEMPTS}): {str(e)}"
            )
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(current_delay)
                current_delay *= 2
    return False


def add_training_sample(features, success, attack_data_samples, attack_labels):
    attack_data_samples.append(features)
    attack_labels.append(1 if success else 0)
    logger.info(f"Добавлен новый обучающий пример с меткой {1 if success else 0}")


def check_signal_outcomes(scraper, pending_targets, attack_data_samples, attack_labels):
    global successful_signals_ever
    targets_to_remove = []
    for target in pending_targets:
        match_id = target["match_id"]
        match_data = get_full_event_data(scraper, match_id)
        if not match_data:
            continue
        home_score = match_data.get("homeScore", {}).get("period1", 0)
        away_score = match_data.get("awayScore", {}).get("period1", 0)
        total_goals_first_half = home_score + away_score

        success = None
        if total_goals_first_half > target.get("goals_at_signal", 0):
            success = True
        else:
            status_type = match_data.get("status", {}).get("type", "").lower()
            current_period = match_data.get("currentPeriod", 0)
            if status_type != "inprogress" or current_period != 1:
                success = False
            else:
                continue

        features = target.get("features", [])
        if features:
            add_training_sample(features, success, attack_data_samples, attack_labels)

        if success:
            message = (
                f"✅ <b>ЦЕЛЬ ПОРАЖЕНА В ПЕРВОМ ТАЙМЕ!</b>\n"
                f"⚽ Матч: {target['match_name']}\n"
                f"📊 Счёт в первом тайме: {home_score} - {away_score}\n"
                f"🎯 Гол забит после сигнала!"
            )
            successful_signals_ever += 1
            send_telegram(message, match_id, is_success_report=True)
        else:
            # Отключено отправление сообщений о неуспешных сигналах
            pass

        print(f"🔔 Исход сигнала для матча {target['match_name']}: {'УСПЕХ' if success else 'НЕ УСПЕХ'}")
        logger.info(f"Цель {match_id} завершена с успехом: {success}")
        targets_to_remove.append(target)

    for t in targets_to_remove:
        pending_targets.remove(t)
    return pending_targets, attack_data_samples, attack_labels


def main():
    global new_signals_in_cycle, total_signals_ever, successful_signals_ever
    pending_targets, sent_notifications, attack_data_samples, attack_labels = load_local_data()
    load_model()
    scraper = cloudscraper.create_scraper(browser="chrome")

    startup_message = "🚀 <b>Бот 'Хищник' запущен и работает! Ожидаются голы в первом тайме!</b> 🦾"
    send_telegram(startup_message)
    print(startup_message)

    while True:
        try:
            new_signals_in_cycle = 0
            first_half_matches = 0

            live_events = get_live_match_events(scraper)
            if not live_events:
                logger.warning("Не удалось получить данные о live-матчах.")
                print("❌ Не удалось получить данные о live-матчах.")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            print(f"🔍 Найдено live матчей: {len(live_events)}")
            print("\n📋 Список матчей в лайве:")
            for idx, event in enumerate(live_events, 1):
                match_id = event.get("id", "N/A")
                home_team = event.get("homeTeam", {}).get("name", "Unknown")
                away_team = event.get("awayTeam", {}).get("name", "Unknown")
                status_desc = event.get("status", {}).get("description", "N/A")
                print(f" {idx}. {home_team} vs {away_team} | ID: {match_id} | Статус: {status_desc}")

            for event_summary in live_events:
                match_id = event_summary["id"]
                if match_id in sent_notifications:
                    continue

                event_data = get_full_event_data(scraper, match_id)
                if not event_data:
                    continue

                tournament_name = event_data.get("tournament", {}).get("name", "").lower()
                if any(keyword in tournament_name for keyword in BLACKLIST_KEYWORDS):
                    continue

                status_type = event_data.get("status", {}).get("type", "").lower()
                current_period = event_data.get("currentPeriod", 0)

                if status_type != "inprogress":
                    continue  # Только матчи в процессе

                stats_data = get_match_statistics(scraper, match_id)
                if not stats_data:
                    logger.warning(f"Статистика для матча {match_id} недоступна, пропускаем.")
                    continue

                incidents_data = get_match_incidents(scraper, match_id)
                features = extract_features(stats_data, event_data)

                triggered1, prob1 = False, 0.0
                if current_period == 1:
                    triggered1, prob1 = strategy_logistic_regression(features, event_data)

                triggered2, prob2 = strategy_shots_corners(features)
                triggered3, prob3 = strategy_possession_attack(features)

                votes = sum([triggered1, triggered2, triggered3])
                triggered = votes >= 2

                probs = [p for tr, p in [(triggered1, prob1), (triggered2, prob2), (triggered3, prob3)] if tr]
                combined_prob = sum(probs) / len(probs) if probs else 0.0

                if triggered:
                    strat_results = [
                        {"name": "RandomForest", "triggered": triggered1, "prob": prob1},
                        {"name": "ShotsCornersRule", "triggered": triggered2, "prob": prob2},
                        {"name": "PossessionAttackRule", "triggered": triggered3, "prob": prob3},
                    ]
                    strat_log = ", ".join(
                        [f"{r['name']}({'+' if r['triggered'] else '-'})({r['prob']:.2f})" for r in strat_results]
                    )
                    message = (
                        f"🔥 <b>Обнаружена цель!</b>\n"
                        f"⚽ Матч: {event_data['homeTeam']['name']} vs {event_data['awayTeam']['name']}\n"
                        f"📊 Счёт: {event_data['homeScore']['current']} - {event_data['awayScore']['current']}\n"
                        f"🎯 Вероятность гола: {combined_prob:.2f}\n"
                        f"⚙️ Стратегии: {strat_log}\n"
                        f"------------------------------------------\n"
                        f"{format_statistics(stats_data, incidents_data)}"
                    )
                    send_telegram(message, match_id)
                    print(
                        f"⚡ Обнаружен сигнал: Матч {match_id} | Вероятность: {combined_prob:.2f} | Стратегии: {strat_log}"
                    )

                    sent_notifications.add(match_id)
                    new_signals_in_cycle += 1
                    total_signals_ever += 1
                    if current_period == 1:
                        first_half_matches += 1

                    pending_targets.append(
                        {
                            "match_id": match_id,
                            "signal_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "features": features,
                            "goals_at_signal": event_data.get("homeScore", {}).get("period1", 0)
                            + event_data.get("awayScore", {}).get("period1", 0),
                            "match_name": f"{event_data['homeTeam']['name']} vs {event_data['awayTeam']['name']}",
                        }
                    )
                else:
                    if current_period == 1:
                        first_half_matches += 1
                    logger.info(
                        f"Матч {match_id} ({event_data['homeTeam']['name']} vs {event_data['awayTeam']['name']}) - без сигнала. Вероятность: {combined_prob:.3f}"
                    )

                time.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))

            pending_targets, attack_data_samples, attack_labels = check_signal_outcomes(
                scraper, pending_targets, attack_data_samples, attack_labels
            )
            train_model(attack_data_samples, attack_labels)

            save_local_data(pending_targets, sent_notifications, attack_data_samples, attack_labels)

            report_msg = (
                f"📊 Отчёт по циклу:\n"
                f"🔍 Найдено матчей live: {len(live_events)}\n"
                f"⏱ Матчей в первом тайме: {first_half_matches}\n"
                f"🔥 Новых сигналов: {new_signals_in_cycle}\n"
                f"✅ Успешных сигналов всего: {successful_signals_ever}\n"
                f"🎯 Всего сигналов всего: {total_signals_ever}"
            )
            print(report_msg)
            logger.info(f"Матчей в первом тайме: {first_half_matches}")
            print(f"⏳ Пауза {CHECK_INTERVAL_SECONDS} секунд перед следующим циклом.\n")
            time.sleep(CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            save_local_data(pending_targets, sent_notifications, attack_data_samples, attack_labels)
            logger.info("Бот остановлен вручную.")
            print("🛑 Бот остановлен вручную.")
            break
        except Exception as e:
            logger.error(f"Критическая ошибка: {str(e)}")
            print(f"❌ Ошибка: {str(e)}. Пауза 60 секунд.")
            time.sleep(60)


if __name__ == "__main__":
    main()


