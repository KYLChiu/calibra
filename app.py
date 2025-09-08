# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# from scipy.stats import norm
# from scipy.optimize import brentq
# import plotly.graph_objects as go


# # -----------------------------
# # Black–Scholes formula with dividend yield q
# # -----------------------------
# def bs_price(S, K, T, r, sigma, option_type="C", q=0.0):
#     d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     if option_type == "C":
#         return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
#     else:
#         return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# # -----------------------------
# # Implied volatility solver
# # -----------------------------
# def implied_vol(price, S, K, T, r, option_type="C", q=0.0):
#     if price <= 0 or T <= 0:
#         return np.nan
#     f = lambda sigma: bs_price(S, K, T, r, sigma, option_type, q) - price
#     try:
#         return brentq(f, 1e-6, 25.0)
#     except Exception:
#         return np.nan


# # -----------------------------
# # Streamlit App
# # -----------------------------
# st.set_page_config(page_title="Vol-Fitter (Black-Scholes)", layout="wide")
# st.title("📊 Vol-Fitter — Black–Scholes Implied Vol Dashboard (Yahoo Finance)")

# # Inputs
# symbol = st.text_input("Ticker", "AAPL")
# right = st.radio("Option Type", ["C", "P"])

# if symbol:
#     # Load ticker and history
#     ticker = yf.Ticker(symbol)
#     _ = ticker.history(period="1d")  # force data fetch
#     spot = ticker.history(period="1d")["Close"].iloc[-1]
#     q = ticker.info.get("dividendYield", 0.0) or 0.0

#     # Expirations
#     expirations = ticker.options
#     if expirations:
#         expiry = st.selectbox("Choose expiry for table", expirations)
#         # Fetch option chain
#         opt_chain = ticker.option_chain(expiry)
#         df_options = opt_chain.calls if right == "C" else opt_chain.puts

#         # Compute mid-price and IV for all strikes
#         T = (pd.to_datetime(expiry) - pd.Timestamp.today()).days / 365
#         rows = []
#         for _, row in df_options.iterrows():
#             if pd.notna(row["lastPrice"]):
#                 price = row["lastPrice"]
#             elif pd.notna(row["bid"]) and pd.notna(row["ask"]):
#                 price = (row["bid"] + row["ask"]) / 2
#             else:
#                 continue
#             iv = implied_vol(price, spot, row["strike"], T, 0.01, right, q)
#             rows.append({"strike": row["strike"], "mid_price": price, "iv": iv})

#         df = pd.DataFrame(rows)
#         st.subheader(f"Option Chain for {symbol} ({expiry}, {right})")
#         st.dataframe(df)

#         # Plot smile (all strikes)
#         fig = go.Figure()
#         fig.add_trace(
#             go.Scatter(
#                 x=df["strike"], y=df["iv"], mode="markers+lines", name="Implied Vol"
#             )
#         )
#         fig.update_layout(
#             title=f"Implied Vol Smile ({symbol}, {expiry}, {right})",
#             xaxis_title="Strike",
#             yaxis_title="IV",
#         )
#         st.plotly_chart(fig, use_container_width=True)

#         # 3D surface for first 3 expiries
#         st.subheader("Vol Surface")
#         surface_rows = []
#         for exp in expirations:
#             T_exp = (pd.to_datetime(exp) - pd.Timestamp.today()).days / 365
#             if T_exp <= 0:
#                 continue
#             opt_chain_exp = ticker.option_chain(exp)
#             df_opt_exp = opt_chain_exp.calls if right == "C" else opt_chain_exp.puts
#             for _, row in df_opt_exp.iterrows():
#                 if pd.notna(row["lastPrice"]):
#                     price = row["lastPrice"]
#                 elif pd.notna(row["bid"]) and pd.notna(row["ask"]):
#                     price = (row["bid"] + row["ask"]) / 2
#                 else:
#                     continue
#                 iv = implied_vol(price, spot, row["strike"], T_exp, 0.01, right, q)
#                 surface_rows.append({"strike": row["strike"], "expiry": exp, "iv": iv})

#         surf_df = pd.DataFrame(surface_rows)
#         if not surf_df.empty:
#             # Pivot to make grid for surface
#             grid = surf_df.pivot(index="expiry", columns="strike", values="iv")
#             X, Y = np.meshgrid(
#                 grid.columns.values,
#                 pd.to_datetime(grid.index).map(
#                     lambda x: (x - pd.Timestamp.today()).days / 365
#                 ),
#             )
#             Z = grid.values
#             fig3d = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis")])
#             fig3d.update_layout(
#                 scene=dict(
#                     xaxis_title="Strike", yaxis_title="Maturity (yrs)", zaxis_title="IV"
#                 )
#             )
#             st.plotly_chart(fig3d, use_container_width=True)
#     else:
#         st.warning("No option expirations available for this symbol.")
