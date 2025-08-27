# Trading Bot - Türkçe Futures Bot

Bu bot, Binance Futures üzerinde otomatik trading yapan gelişmiş bir sistemdir. Machine Learning tabanlı sinyal üretimi, dinamik TP/SL yönetimi ve güvenli pozisyon boyutlandırması ile çalışır.

## Özellikler

- 🤖 **ML Tabanlı Sinyal Üretimi**: RandomForest sınıflandırma ile akıllı sinyal analizi
- 📊 **Dinamik TP/SL Yönetimi**: ATR ve olasılık bazlı otomatik TP/SL hesaplama
- 📈 **Trailing Stop**: Kar kilitleme için otomatik trailing stop desteği
- 🔒 **Risk Yönetimi**: Hesap sermayesi bazlı pozisyon boyutlandırma
- 📱 **Telegram Entegrasyonu**: Anlık bildirimler ve durum raporları
- 🎯 **Gelişmiş Filtreler**: HTF trend ve yumuşak giriş filtreleri

## Kurulum

1. **Bağımlılıkları yükleyin:**
```bash
pip install asyncio aiohttp numpy pandas talib binance python-dotenv scikit-learn joblib
```

2. **Yapılandırma dosyasını oluşturun:**
```bash
cp .env.example .env
```

3. **.env dosyasını düzenleyin:**
- Binance API anahtarlarınızı ekleyin
- Telegram bot token ve chat ID'sini ayarlayın
- Diğer parametreleri ihtiyacınıza göre ayarlayın

## Gelişmiş Giriş Filtresi ve HTF Trend Filtresi

### HTF (Yüksek Zaman Dilimi) Trend Filtresi

HTF trend filtresi, daha büyük zaman dilimlerinden trend yönünü analiz ederek sinyal kalitesini artırır:

```env
# HTF trend filtresini aktif et
USE_HTF_TREND=true

# 1 saatlik timeframe kullan (varsayılan)
HTF_TREND_TF=1h

# EMA200 ile trend belirle (varsayılan) 
HTF_TREND_EMA=200

# %0.05 tolerans (5 bp)
HTF_TREND_TOL_BP=5.0
```

**Çalışma Mantığı:**
- **UP trend**: Sadece LONG sinyalleri alınır
- **DOWN trend**: Sadece SHORT sinyalleri alınır  
- **NONE trend**: Hiçbir sinyal alınmaz
- **Devre dışı**: Tüm sinyaller normal şekilde değerlendirilir

### Yumuşak Giriş Filtreleri

Aşırı uzamış fiyatlardan kaçınmak ve pullback beklemek için:

```env
# 5 dakikalık timeframe'de filtre uygula
ENTRY_TF=5m

# EMA20'den maksimum %1.5 sapma
EXT_MAX_DEV_PCT=1.5

# Son 10 bara bak
PULLBACK_LOOKBACK=10

# En az 2 bar EMA temas
MIN_PULLBACK_BARS=2

# Spike korunmasını aktif et
USE_SPIKE_GUARD=true

# 12 bar spike kontrolü (~1 saat)
SPIKE_LOOKBACK=12

# Maksimum %5 spike hareketi
SPIKE_MAX_PCT=5.0
```

### Pozisyon Boyutlandırma Geliştirmeleri

#### Taban Marj Hedefi Modu
```env
# 6 USDT marj hedefle, 3x kaldıraçla ~18 USDT notional
TARGET_MARGIN_USDT=6.0
DEFAULT_LEVERAGE=3
```

#### Yüzde Modu
```env
POSITION_SIZING_MODE=percent
POSITION_PERCENT=5.0  # Hesap bakiyesinin %5'i
```

#### Risk Modu (Varsayılan)
```env
POSITION_SIZING_MODE=risk
MAX_ACCOUNT_RISK_PERCENT=2.0  # ATR bazlı risk
```

### Pozisyon Limitleri

Yön bazlı pozisyon limitleri ile risk kontrolü:

```env
MAX_LONG_POSITIONS=7   # Maksimum long pozisyon sayısı
MAX_SHORT_POSITIONS=7  # Maksimum short pozisyon sayısı
```

## Kullanım

```bash
python scanner.py
```

Bot başladığında:
1. Model dosyalarını yükler/eğitir
2. Telegram'a başlangıç bildirimi gönderir
3. Sembol taramasına başlar
4. Sinyalleri değerlendirir ve pozisyon açar
5. Açık pozisyonları sürekli izler

## Telegram Komutları

Bot şu durumlarda otomatik bildirim gönderir:
- 🟢 Yeni pozisyon açıldığında
- 🎯 TP seviyesi vurulduğunda  
- 🔴 SL seviyesi vurulduğunda
- ⚠️ ML tabanlı kapatma yapıldığında
- 📊 Periyodik PnL raporları

## Güvenlik ve Risk Yönetimi

- **Maksimum Risk**: Hesap bakiyesinin %2'si (değiştirilebilir)
- **Pozisyon Limitleri**: Toplam ve yön bazlı limitler
- **ML Koruması**: Düşük olasılık pozisyonlarını otomatik kapatma
- **Trailing Stop**: Kar kilitleme mekanizması
- **Break-even Koruması**: TP1 sonrası zarar önleme

## Dosya Yapısı

- `scanner.py` - Ana bot dosyası
- `.env` - Yapılandırma dosyası
- `positions.json` - Aktif pozisyonlar
- `history_reinforced.json` - İşlem geçmişi
- `model_cls.pkl` - ML sınıflandırma modeli
- `model_meta.json` - Model meta verileri

## Sorun Giderme

1. **API Bağlantı Hatası**: API anahtarlarını ve izinleri kontrol edin
2. **Telegram Bildirimi Çalışmıyor**: Bot token ve chat ID'yi doğrulayın
3. **Model Yüklenmiyor**: İlk çalıştırmada model otomatik oluşturulur
4. **Pozisyon Açılmıyor**: Bakiye, limit ve sinyal parametrelerini kontrol edin

## Dikkat Edilecekler

⚠️ **ÖNEMLİ**: Bu bot gerçek para ile çalışır. Test ortamında deneyim kazandıktan sonra küçük miktarlarla başlayın.

⚠️ **Risk Uyarısı**: Kripto para ticareti yüksek risk içerir. Kaybetmeyi göze alamayacağınız para ile işlem yapmayın.

⚠️ **Gözlem**: Botu sürekli izleyin ve anormal durumları kontrol edin.

## Sürüm Notları

**v5.1**
- Stabil TP/SL yönetimi
- ML tabanlı kapatma korumaları
- HTF trend filtresi
- Yumuşak giriş filtreleri
- Taban marj hedefi desteği
- Yön bazlı pozisyon limitleri

---

**Geliştirici**: Bu bot eğitim amaçlıdır. Kullanımından doğacak zararlardan kullanıcı sorumludur.