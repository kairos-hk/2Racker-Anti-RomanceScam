import os
import json
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QMessageBox, QSystemTrayIcon, QMenu, QAction,
    QTabWidget, QLineEdit, QSpinBox, QProgressBar, QTextEdit, QHBoxLayout,
    QScrollArea
)
from PyQt5.QtGui import (
    QIcon, QPixmap, QPainter, QColor, QCursor, QGuiApplication, QFont
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from telethon.sync import TelegramClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
api_id = int(os.getenv('api_id'))
api_hash = os.getenv('api_hash')
LOG_FILE = 'scan_log.json'
LAST_SCAN_FILE = 'last_scan.json'
_client_instance = None

model_path = "./koelectra-romance-scam"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

class AnalysisThread(QThread):
    result_signal = pyqtSignal(str, str)

    def __init__(self, name, texts):
        super().__init__()
        self.name = name
        self.texts = texts

    def run(self):
        joined = ' [SEP] '.join(self.texts)
        label = predict_romance_scam(joined)
        self.result_signal.emit(self.name, label)

def get_client():
    global _client_instance
    if _client_instance is None:
        _client_instance = TelegramClient('scamdetect_session', api_id, api_hash)
        _client_instance.connect()
    return _client_instance

def predict_romance_scam(text: str):
    try:
        model.eval()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
        label = "로맨스 스캠" if pred == 1 else "정상 대화"
        return label
    except Exception as e:
        return f'오류 발생: {str(e)}'

def save_log(result):
    logs = load_logs()
    logs.insert(0, result)
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(logs[:100], f, indent=2, ensure_ascii=False)

def load_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_last_scan(name):
    data = load_last_scans()
    data[name] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LAST_SCAN_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_last_scans():
    if os.path.exists(LAST_SCAN_FILE):
        with open(LAST_SCAN_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

class ScamDetectPopup(QWidget):
    def __init__(self, tray_icon):
        super().__init__(flags=Qt.Popup)
        self.setWindowTitle('2Racker 로맨스 스캠 탐지기')
        self.setMinimumSize(900, 650)
        self.tray_icon = tray_icon
        self.setStyleSheet('''
            QWidget {
                font-family: 'Segoe UI', 'Noto Sans KR';
                color: #ffffff;
                background-color: #1a252f;
            }
            QTabWidget::pane {
                border: 1px solid #2d3b4e;
                background: #222d3a;
            }
            QTabBar::tab {
                background: #2d3b4e;
                color: #a0b3c5;
                border: none;
                padding: 12px 20px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background: #0078d4;
                color: #ffffff;
            }
            QPushButton {
                background-color: #0078d4;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #005ea2;
            }
            QLineEdit, QSpinBox {
                background-color: #2d3b4e;
                color: #ffffff;
                border: 1px solid #3e4b5e;
                border-radius: 4px;
                padding: 8px;
            }
            QLineEdit:focus, QSpinBox:focus {
                border: 1px solid #0078d4;
            }
            QListWidget {
                background-color: #2d3b4e;
                border: none;
                color: #ffffff;
            }
            QListWidget::item:selected {
                background-color: #3e4b5e;
            }
            QProgressBar {
                text-align: center;
                height: 20px;
                border: 1px solid #3e4b5e;
                border-radius: 10px;
                background: #2d3b4e;
                color: #ffffff;
            }
            QProgressBar::chunk {
                border-radius: 9px;
                background-color: #0078d4;
            }
            QTextEdit {
                background-color: #2d3b4e;
                color: #ffffff;
                border: 1px solid #3e4b5e;
                border-radius: 4px;
                padding: 8px;
            }
            QLabel {
                color: #a0b3c5;
            }
        ''')

        self.tabs = QTabWidget()
        self.login_tab = QWidget()
        self.status_tab = QWidget()
        self.logs_tab = QWidget()
        self.settings_tab = QWidget()

        self.scan_interval = 5
        self.analysis_threads = []
        self.privacy_agreed = False

        self.init_login_tab()
        self.init_status_tab()
        self.init_logs_tab()
        self.init_settings_tab()

        self.tabs.addTab(self.login_tab, '인증')
        self.tabs.addTab(self.status_tab, '분석 현황')
        self.tabs.addTab(self.logs_tab, '분석 기록')
        self.tabs.addTab(self.settings_tab, '설정')
        self.tabs.currentChanged.connect(self.on_tab_changed)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.auto_scan_selected_chats)
        self.timer.start(self.scan_interval * 60 * 1000)

        if not os.path.exists('scamdetect_session.session'):
            self.tabs.setCurrentWidget(self.login_tab)
        else:
            client = get_client()
            if not client.is_user_authorized():
                self.tabs.setCurrentWidget(self.login_tab)
            else:
                self.privacy_agreed = True
                self.tabs.setCurrentWidget(self.status_tab)

    def init_login_tab(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)
        header_label = QLabel('2Racker 로맨스 스캠 탐지기')
        header_label.setStyleSheet('font-size: 18px; font-weight: bold; color: #ffffff;')
        layout.addWidget(header_label)
        auth_layout = QVBoxLayout()
        auth_layout.setSpacing(6)
        
        self.phone_input = QLineEdit()
        self.phone_input.setPlaceholderText('전화번호 입력 (예: +821012345678)')
        auth_layout.addWidget(self.phone_input)
        
        self.code_input = QLineEdit()
        self.code_input.setPlaceholderText('인증 코드 입력')
        auth_layout.addWidget(self.code_input)
        
        button_layout = QHBoxLayout()
        code_btn = QPushButton('인증 코드 요청')
        code_btn.clicked.connect(self.send_code)
        button_layout.addWidget(code_btn)
        
        login_btn = QPushButton('로그인')
        login_btn.clicked.connect(self.sign_in)
        button_layout.addWidget(login_btn)
        
        auth_layout.addLayout(button_layout)
        layout.addLayout(auth_layout)
        
        privacy_label = QLabel('개인정보 처리방침')
        privacy_label.setStyleSheet('font-size: 14px; font-weight: bold; color: #ffffff;')
        layout.addWidget(privacy_label)
        
        privacy_text = QTextEdit()
        privacy_text.setReadOnly(True)
        privacy_text.setText("""
<2Racker 로맨스 스캠 탐지기 개인정보 처리방침>
                             
당사는 사용자의 개인정보 보호를 최우선으로 고려하며, 개인정보보호법에 의거하여 다음과 같은 방침을 준수합니다.
서비스 이용을 위해 아래 동의 버튼을 클릭해 주세요.

[제 1 장 총칙]
제 1 조 (목적)
이 약관은 2Racker(이하 “atorial사”)가 제공하는 “2Racker 로맨스 스캠 탐지기”(이하 “서비스”)가 Telegram으로부터 제공받은 대화 데이터를
수집·이용함에 있어 이용자의 개인정보 보호 및 권리·의무, 책임사항 등을 규정함을 목적으로 한다.

제 2 조 (약관의 효력 및 변경)
① 본 약관은 이용자가 동의하는 시점부터 효력을 발생한다.
② 운영사는 관련 법령 및 정책 변경, 서비스 개선을 위해 본 약관을 변경할 수 있으며, 변경 시 서비스 내 공지 또는 개별 통지한다.
③ 이용자는 변경된 약관에 동의하지 않을 경우 서비스 이용을 중단하고, 동의 시 변경된 약관에 대한 동의로 간주된다.

제 3 조 (용어의 정의)
이 약관에서 사용하는 용어의 정의는 다음과 같다.
“서비스”라 함은 운영사가 제공하는 “2Racker 로맨스 스캠 탐지기” 애플리케이션 및 관련 기능 일체를 의미한다.
“이용자”라 함은 본 약관에 동의하고 서비스를 이용하는 자를 의미한다.
“Telegram 제공 데이터”라 함은 이용자가 Telegram 사로부터 운영사가 제공받는 모든 대화 메시지 및 식별 정보를 의미한다.
“개인정보”라 함은 생존하는 개인을 식별할 수 있는 정보로서, Telegram 제공 데이터 중 이용자명·ID·메시지 내용을 포함한다.

[제 2 장 개인정보의 수집 및 이용]
제 4 조 (개인정보의 수집 주체)
본 서비스는 Telegram이라는 외부 주체로부터 API를 통해 이용자의 대화 데이터를 제공받아 처리한다.
                             
제 5 조 (수집·이용 목적)
운영사는 다음의 목적으로 Telegram 제공 데이터를 수집·이용한다.
- 로맨스스캠 의심 대화 탐지 및 알림 제공
- 분석 이력 및 로그 관리

제 6 조 (수집하는 개인정보 항목)
Telegram 사용자 식별정보(이용자명, 사용자 ID)
Telegram 채팅 메시지 텍스트

제 7 조 (개인정보 보유·이용 기간)
Telegram 제공 데이터는 분석 완료 즉시 처리되며, 분석 로그는 이용자 기기에 보관 후 자동 삭제한다.
이용자가 서비스 이용을 중단하거나 동의를 철회하면, 보유 중인 모든 개인정보는 지체 없이 파기한다.

제 8 조 (제3자 제공 및 위탁)
운영사는 Telegram으로부터 제공받은 데이터를 외부 기관, 제3자 또는 위탁 업체에 제공하거나 위탁하지 않는다.

[제 3 장 정보주체의 권리·의무 및 행사]
제 9 조 (정보주체의 권리)
이용자는 언제든지 아래 권리를 행사할 수 있다.
- 개인정보 열람 요구
- 오류 등이 있을 경우 정정 요구
- 삭제(파기) 요구
- 처리정지 요구
- 동의 철회

제 10 조 (권리 행사 방법)
요청 방법: 이메일 또는 우편
이메일: kairos.hk0912@snoo-py.org
주소: 경상북도 의성군 봉양면 봉호로 14, 2Racker
운영사는 지체 없이 조치하고 그 결과를 통지한다.

[제 4 장 안전성 확보 조치]
제 11 조 (기술적·관리적 보호조치)
내부관리계획 수립 및 시행
개인정보 처리 시스템 접근통제
주기적 보안 교육 및 점검 시행

[제 5 장 기타]
제 12 조 (개인정보관리 책임자)
성명: 김동영
소속: 2Racker
연락처: kairos.hk0912@snoo-py.org

제 13 조 (동의 거부 시 불이익)
본 약관에 동의하지 않을 경우 “2Racker 로맨스 스캠 탐지기”의 대화 분석 기능을 이용할 수 없다.
        """)
        scroll_area = QScrollArea()
        scroll_area.setWidget(privacy_text)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedHeight(200)
        layout.addWidget(scroll_area)
        
        agree_btn = QPushButton('개인정보 처리방침 동의')
        agree_btn.clicked.connect(self.agree_privacy)
        layout.addWidget(agree_btn)
        
        layout.addStretch()
        self.login_tab.setLayout(layout)

    def agree_privacy(self):
        self.privacy_agreed = True
        QMessageBox.information(self, '동의 완료', '개인정보 처리방침에 동의하셨습니다.')

    def send_code(self):
        try:
            self.client = TelegramClient('scamdetect_session', api_id, api_hash)
            self.client.connect()
            phone = self.phone_input.text().strip()
            self.client.send_code_request(phone)
            QMessageBox.information(self, '코드 전송', 'Telegram 앱으로 인증 코드가 전송되었습니다.')
        except Exception as e:
            QMessageBox.critical(self, '에러', f'코드 전송 실패: {str(e)}')

    def sign_in(self):
        if not self.privacy_agreed:
            QMessageBox.warning(self, '안내', '개인정보 처리방침에 동의해야 로그인이 가능합니다.')
            return
        
        phone = self.phone_input.text().strip()
        code = self.code_input.text().strip()
        try:
            self.client.sign_in(phone, code)
            global _client_instance
            _client_instance = self.client
            self.current_phone = phone
            QMessageBox.information(self, '성공', '로그인 완료!')
            self.populate_chat_list()
            self.update_account_info()
            self.tabs.setCurrentWidget(self.status_tab)
        except Exception as e:
            QMessageBox.critical(self, '로그인 실패', str(e))

    def init_status_tab(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        header_label = QLabel('분석 현황')
        header_label.setStyleSheet('font-size: 16px; font-weight: bold; color: #ffffff;')
        layout.addWidget(header_label)
        
        self.status_label = QLabel('AI가 채팅을 분석 중입니다...')
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.chat_list = QListWidget()
        self.chat_list.itemClicked.connect(lambda item: self.analyze_single_chat(item))
        layout.addWidget(QLabel('분석 대상 채팅'))
        layout.addWidget(self.chat_list)
        
        self.scan_button = QPushButton('전체 채팅 분석')
        self.scan_button.clicked.connect(self.run_full_scan)
        layout.addWidget(self.scan_button)
        
        self.status_tab.setLayout(layout)
        self.populate_chat_list()

    def populate_chat_list(self):
        self.chat_list.clear()
        client = get_client()
        try:
            dialogs = client.get_dialogs()
            last_scans = load_last_scans()
            for dialog in dialogs:
                if dialog.is_user:
                    item = QListWidgetItem(dialog.name)
                    if dialog.name in last_scans:
                        item.setToolTip(f'마지막 분석: {last_scans[dialog.name]}')
                    self.chat_list.addItem(item)
        except Exception:
            pass

    def analyze_single_chat(self, item):
        name = item.text()
        self.analyze_chat(name)

    def analyze_chat(self, name):
        client = get_client()
        try:
            dialog = next((d for d in client.get_dialogs() if d.name == name), None)
            if not dialog:
                return
            messages = client.iter_messages(dialog.entity, limit=10)
            texts = [msg.text.strip() for msg in messages if msg.text]
            if not texts:
                return
            
            analysis_thread = AnalysisThread(name, texts)
            analysis_thread.result_signal.connect(self.handle_analysis_result)
            self.analysis_threads.append(analysis_thread)
            analysis_thread.start()
            
        except Exception as e:
            self.status_label.setText(f'오류 발생: {str(e)}')

    def handle_analysis_result(self, name, label):
        save_last_scan(name)
        log = {'user': name, 'result': label, 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        save_log(log)
        self.status_label.setText(f"'{name}' 분석 완료 → {label}")
        self.refresh_logs()
        if label == '로맨스 스캠':
            self.tray_icon.showMessage('스캠 감지!', f'Telegram의 {name} 채팅에서 로맨스 스캠이 감지되었습니다.', QSystemTrayIcon.Critical)

    def auto_scan_selected_chats(self):
        self.run_full_scan()

    def run_full_scan(self):
        client = get_client()
        try:
            dialogs = client.get_dialogs()
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(len([d for d in dialogs if d.is_user]))
            self.progress_bar.setValue(0)
            
            for dialog in dialogs:
                if not dialog.is_user:
                    continue
                self.analyze_chat(dialog.name)
                self.progress_bar.setValue(self.progress_bar.value() + 1)
            
            self.progress_bar.setVisible(False)
            self.status_label.setText('전체 분석 완료')
        except Exception:
            self.progress_bar.setVisible(False)
            self.status_label.setText('분석 중 오류 발생')

    def init_logs_tab(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        header_label = QLabel('분석 기록')
        header_label.setStyleSheet('font-size: 16px; font-weight: bold; color: #ffffff;')
        layout.addWidget(header_label)
        
        self.log_list = QListWidget()
        layout.addWidget(self.log_list)
        self.logs_tab.setLayout(layout)
        self.refresh_logs()

    def refresh_logs(self):
        self.log_list.clear()
        logs = load_logs()
        for log in logs:
            text = f"[{log['time']}] {log['user']} → {log['result']}"
            item = QListWidgetItem(text)
            item.setForeground(QColor('#ff4d4f') if log['result'] == '로맨스 스캠' else QColor('#4caf50'))
            self.log_list.addItem(item)

    def init_settings_tab(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        header_label = QLabel('시스템 설정')
        header_label.setStyleSheet('font-size: 16px; font-weight: bold; color: #ffffff;')
        layout.addWidget(header_label)
        
        layout.addWidget(QLabel('계정 정보'))
        self.account_label = QLabel()
        layout.addWidget(self.account_label)
        
        logout_btn = QPushButton('로그아웃 및 종료')
        logout_btn.clicked.connect(self.logout)
        layout.addWidget(logout_btn)

        layout.addWidget(QLabel('자동 분석 주기 (분)'))
        self.interval_spin = QSpinBox()
        self.interval_spin.setMinimum(1)
        self.interval_spin.setValue(self.scan_interval)
        self.interval_spin.valueChanged.connect(self.update_interval)
        layout.addWidget(self.interval_spin)

        layout.addStretch()
        self.settings_tab.setLayout(layout)
        self.update_account_info()

    def update_account_info(self):
        try:
            client = get_client()
            if client.is_user_authorized():
                me = client.get_me()
                name = f"{me.first_name or ''} {me.last_name or ''}".strip()
                username = f"(@{me.username})" if me.username else ''
                phone = getattr(me, 'phone', self.current_phone if hasattr(self, 'current_phone') else '')
                info = f"이름: {name}\n전화번호: {phone} {username}"
            else:
                info = "로그인 필요"
        except Exception:
            info = "계정 정보를 불러올 수 없습니다."
        self.account_label.setText(info)

    def logout(self):
        try:
            client = get_client()
            client.log_out()
            client.disconnect()
        except Exception:
            pass
        global _client_instance
        _client_instance = None
        try:
            os.remove('scamdetect_session.session')
            os.remove(LOG_FILE)
            os.remove(LAST_SCAN_FILE)
        except Exception:
            pass
        QApplication.quit()

    def update_interval(self, value):
        self.scan_interval = value
        self.timer.stop()
        self.timer.start(self.scan_interval * 60 * 1000)

    def on_tab_changed(self, index):
        if self.tabs.widget(index) == self.settings_tab:
            self.update_account_info()

    def show_at_cursor(self):
        self.adjustSize()
        cursor_pos = QCursor.pos()
        screen_geometry = QGuiApplication.primaryScreen().availableGeometry()
        x = min(cursor_pos.x(), screen_geometry.width() - self.width())
        y = min(cursor_pos.y(), screen_geometry.height() - self.height())
        self.move(x, y)
        self.show()

def generate_tray_icon():
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor('#0078d4'))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(4, 4, 24, 24)
    painter.setPen(QColor('#ffffff'))
    painter.setFont(QFont('Segoe UI', 16, QFont.Bold))
    painter.end()
    return QIcon(pixmap)

def launch_app():
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    app.setFont(QFont('Segoe UI', 10))
    app.setStyleSheet('''
        QWidget { background-color: #1a252f; }
    ''')

    icon = generate_tray_icon()
    tray_icon = QSystemTrayIcon(icon)
    tray_icon.setToolTip('2Racker 로맨스 스캠 탐지기')

    popup = ScamDetectPopup(tray_icon)

    tray_menu = QMenu()
    open_action = QAction('시스템 열기')
    open_action.triggered.connect(popup.show_at_cursor)
    tray_menu.addAction(open_action)

    quit_action = QAction('종료')
    quit_action.triggered.connect(app.quit)
    tray_menu.addAction(quit_action)

    tray_icon.setContextMenu(tray_menu)
    tray_icon.activated.connect(lambda reason: popup.show_at_cursor() if reason == QSystemTrayIcon.Trigger else None)
    tray_icon.show()

    return app.exec_()

if __name__ == '__main__':
    launch_app()