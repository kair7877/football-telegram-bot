import json
import time
import os
from typing import Any, Dict, List, Set
import requests
import cloudscraper

# Получаем токен и чат ID из переменных окружения
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

# --- НАСТРОЙКИ ---
SHOTS_ON_TARGET_THRESHOLD = 4
CHECK_INTERVAL_SECONDS = 120
BASE_URL = "https://api.sofascore.com/api/v1/"


def send_telegram_notification(message: str):
    """Отправляет сообщение в Telegram."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    params = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            print("✅ Уведомление отправлено.")
        else:
            print(f"❌ Ошибка Telegram API: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"⚠️ Ошибка подключения к Telegram: {e}")


def get_live_match_events(scraper: cloudscraper.CloudScraper) -> List[Dict[str, Any]]:
    try:
        url = f"{BASE_URL}sport/football/events/live"
        response = scraper.get(url)
        response.raise_for_status()
        return response.json().get("events", [])
    except Exception as e:
        print(f"⚠️ Ошибка получения live-матчей: {e}")
        return []


def get_match_statistics(scraper: cloudscraper.CloudScraper, match_id: int) -> Dict[str, Any]:
    try:
        url = f"{BASE_URL}event/{match_id}/statistics"
        response = scraper.get(url)
        response.raise_for_status()
        stats_all = response.json().get("statistics", [])
        if stats_all:
            return stats_all[0]
    except Exception:
        pass
    return {}


def check_matches_for_shots(scraper: cloudscraper.CloudScraper, sent_notifications: Set[str]):
    print(f"🔄 Проверка в {time.strftime('%H:%M:%S')}")
    live_events = get_live_match_events(scraper)

    if not live_events:
        print("⛔ Нет активных матчей.")
        return

    for event in live_events:
        try:
            home_team = event['homeTeam']['name']
            away_team = event['awayTeam']['name']
            match_id = event['id']
            stats_data = get_match_statistics(scraper, match_id)

            if not stats_data:
                continue

            for group in stats_data.get("groups", []):
                for item in group.get("statisticsItems", []):
                    if item.get("name") == "Shots on target":
                        home_shots = int(item["home"])
                        away_shots = int(item["away"])

                        if home_shots >= SHOTS_ON_TARGET_THRESHOLD:
                            notify_id = f"{match_id}-{home_team}"
                            if notify_id not in sent_notifications:
                                msg = f"🎯 <b>{home_team}</b> достиг {home_shots} ударов в створ!\n🏆 {event['tournament']['name']}\n⚽️ {home_team} vs {away_team}\n📈 {home_shots} - {away_shots}"
                                send_telegram_notification(msg)
                                sent_notifications.add(notify_id)

                        if away_shots >= SHOTS_ON_TARGET_THRESHOLD:
                            notify_id = f"{match_id}-{away_team}"
                            if notify_id not in sent_notifications:
                                msg = f"🎯 <b>{away_team}</b> достиг {away_shots} ударов в створ!\n🏆 {event['tournament']['name']}\n⚽️ {home_team} vs {away_team}\n📈 {home_shots} - {away_shots}"
                                send_telegram_notification(msg)
                                sent_notifications.add(notify_id)
                        return  # только первый матч в этом цикле
        except Exception:
            continue


def main():
    print("🚀 Бот запущен!")
    send_telegram_notification("✅ <b>Бот успешно запущен!</b>\nСледим за ударами в створ...")

    scraper = cloudscraper.create_scraper()
    sent_notifications = set()

    while True:
        try:
            check_matches_for_shots(scraper, sent_notifications)
            time.sleep(CHECK_INTERVAL_SECONDS)
        except Exception as e:
            print(f"❌ Ошибка: {e}. Перезапуск через 60 сек.")
            time.sleep(60)


if __name__ == '__main__':
    main()
