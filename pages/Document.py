import streamlit as st

st.title("利用方法")
st.subheader("はじめに")
st.write("ベイズ最適化(Bayesian Optimization)は、機械学習の分野では"
         "ハイパーパラメータチューニングに利用されることが多い"
         "ブラックボックス関数の最適化手法の一つです。"
         "ベイズ最適化は材料開発においても注目されており、"
         "配合設計やプロセス条件の最適化を高速に実現できます。"
         "このアプリケーションは、ノーコードでベイズ最適化により次の実験候補を提案するツールです。"
         "密度と引張強度、透過率のような複数の要件を満たす必要がある場合でも利用可能です。"
         "ベイズ最適化➡実際に実験➡データセットに追加➡ベイズ最適化...というサイクルを繰り返す"
         "ことで効率的に実験を行うことができます。"
         )

st.subheader("1. Dataset")
st.write("事前に得られているデータセット(実験結果や計算結果)をアップロードします。"
         "説明変数と目的変数を含むcsvファイルをアップロードしてください。")
st.write("例）")
st.write("説明変数：材料Aの配合比、材料Bの配合比、温度条件、圧力...のような実験条件")
st.write("目的変数：密度、引張強度、透過率...のような特性")

st.subheader("2. Setting of X")
st.write("説明変数の設定を行います。説明変数の列名を選択してください。"
         "選択後、各説明変数について上限値、下限値を設定してください。")

st.subheader("3. Setting of Y")
st.write("目的変数の設定を行います。目的変数の列名を選択してください。"
         "選択後、各目的変数について以下の選択肢から要件に合うように選んでください。")
st.write("Maximization : 目的変数の最大値を探索します。")
st.write("Minimization : 目的変数の最小値を探索します。")
st.write("Range : 目的変数がある範囲に収まるような条件を探索します。"
         "この設定を利用する場合には範囲を指定してください。")

st.subheader("4. Acquisition function")
st.write("獲得関数(Acquisition function)の設定を行います。"
         "Probability of improvement(PI)はExpected improvement(EI)に比べ"
         "極小値にトラップされやすいですが、目的関数が複数の場合には取り扱いが容易なため"
         "現在はPIのみを利用可能です。")

st.subheader("5. Get the next condition")
st.write("ベイズ最適化により次の実験条件を取得します。"
         "Performをクリックし、しばらくすると'Next condition'として出力されます。")


