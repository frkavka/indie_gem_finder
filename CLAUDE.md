# indie-gem-finder

Steamのインディーゲームレコメンダー。A/Bテスト型SwipeUIと、TF-IDF + BERTのデュアル空間マッチングを組み合わせたシステム。

## プロジェクト構成

- `app.py` — Streamlit A/B UIアプリ（`recommendation_base.csv` を読み込んで起動）
- `org.py` — オフラインデータパイプライン（Steam/SteamSpy APIからベクトルを生成しCSVに出力）
- `games.csv` — Kaggle Steamゲームカタログ（約12万件）

## アプリの起動

```bash
streamlit run app.py
```

## パイプラインの流れ

1. `org.py` を実行 → `recommendation_base.csv` を生成
2. `streamlit run app.py` でUIを起動
3. UI上でナシ判定したAppIDを `org.py` の `rejected_appids` に追記して再実行（V4直交射影）

## 利用可能なスキル

| コマンド | 説明 |
|---------|------|
| `/sdd-req100 <slug>` | EARS準拠の要件定義を生成・採点（C.U.T.E.スコア >= 98目標） |
| `/sdd-design <slug>` | C4モデル + Arc42 アーキテクチャ設計書を生成 |
| `/sdd-tasks <slug>` | Kiro形式タスク分解（6フェーズ + Ganttチャート） |
| `/sdd-threat <slug>` | STRIDE脅威モデルを生成 |
| `/sdd-full <slug>` | 上記7成果物を依存順に一括生成（L3 Implementation Ready目標） |

スペックの出力先はデフォルトで `.kiro/specs/<slug>/` に保存されます。
