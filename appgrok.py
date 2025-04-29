import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
from google.cloud import bigquery
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from scipy.interpolate import make_interp_spline

# Thiết lập xác thực Google BigQuery
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/CÁ NHÂN/terminal/terminal/etl-cap3-27b899b6d343.json"
client = bigquery.Client(project='etl-cap3')

# Thiết lập tiêu đề và mô tả
st.title("Ứng Dụng Phân Tích Doanh Thu và Phân Cụm Khách Hàng")
st.markdown("""
Ứng dụng này hiển thị phân cụm khách hàng, dự đoán doanh thu, và các biểu đồ phân tích dựa trên dữ liệu từ Google BigQuery.
Vui lòng chọn tab để xem các phân tích chi tiết.
""")

# Tải dữ liệu từ BigQuery
@st.cache_data
def load_data():
    query = """
        SELECT * FROM etl-cap3.Sale_AMZ_ETSY.FinalData
        LIMIT 500000000
    """
    df = client.query(query).to_dataframe()
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['year'] = df['Order Date'].dt.year
    return df

# Tải dữ liệu
with st.spinner("Đang tải dữ liệu từ BigQuery..."):
    df = load_data()

# Tạo các tab
tab1, tab2, tab3, tab4 = st.tabs(["Tổng Quan Doanh Thu", "Dự Đoán Doanh Thu", "Phân Cụm Khách Hàng", "Đánh Giá Mô Hình"])

# Tab 1: Tổng Quan Doanh Thu
with tab1:
    st.header("Tổng Quan Doanh Thu Theo Năm")
    revenue_by_year = df.groupby('year')['Order Total'].sum().reset_index()

    # Vẽ biểu đồ doanh thu theo năm với Plotly
    fig = px.bar(revenue_by_year, x='year', y='Order Total', title='Tổng Doanh Thu Theo Năm',
                 labels={'year': 'Năm', 'Order Total': 'Tổng Doanh Thu'}, color_discrete_sequence=['red'])
    fig.update_layout(xaxis_tickangle=0, yaxis=dict(griddash='dash', gridcolor='gray'))
    st.plotly_chart(fig)

# Tab 2: Dự Đoán Doanh Thu
with tab2:
    st.header("Dự Đoán Doanh Thu với Prophet")

    # Chuẩn bị dữ liệu cho Prophet
    prophet_df = df[['Order Date', 'Order Total']].rename(columns={'Order Date': 'ds', 'Order Total': 'y'})
    prophet_df = prophet_df.groupby('ds').sum().reset_index()

    # Làm mượt dữ liệu thực tế
    prophet_df['y_smooth'] = prophet_df['y'].rolling(window=5, center=True, min_periods=1).mean()
    prophet_df['ds_numeric'] = prophet_df['ds'].apply(lambda x: x.timestamp())
    prophet_df = prophet_df.sort_values('ds_numeric')

    # Nội suy để làm mượt
    x = prophet_df['ds_numeric']
    y = prophet_df['y_smooth']
    x_smooth = np.linspace(x.min(), x.max(), 500)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_smooth)
    ds_smooth = pd.to_datetime(x_smooth, unit='s')

    # Huấn luyện mô hình Prophet
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, changepoint_prior_scale=0.01)
    model.fit(prophet_df)

    # Biểu đồ 1: Thực tế và dự đoán trong phạm vi dữ liệu gốc
    past_future = prophet_df[['ds']].copy()
    past_forecast = model.predict(past_future)
    past_forecast['yhat_smooth'] = past_forecast['yhat'].rolling(window=5, center=True, min_periods=1).mean()
    past_forecast['yhat_lower_smooth'] = past_forecast['yhat_lower'].rolling(window=5, center=True, min_periods=1).mean()
    past_forecast['yhat_upper_smooth'] = past_forecast['yhat_upper'].rolling(window=5, center=True, min_periods=1).mean()

    # Vẽ biểu đồ với Plotly
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=ds_smooth, y=y_smooth, mode='lines', name='Thực tế', line=dict(color='blue', width=1)))
    fig1.add_trace(go.Scatter(x=past_forecast['ds'], y=past_forecast['yhat_smooth'], mode='lines', name='Dự đoán', line=dict(color='orange', width=1)))
    fig1.add_trace(go.Scatter(x=past_forecast['ds'], y=past_forecast['yhat_upper_smooth'], mode='lines', name='Khoảng tin cậy (trên)', line=dict(color='yellow', width=0), showlegend=False))
    fig1.add_trace(go.Scatter(x=past_forecast['ds'], y=past_forecast['yhat_lower_smooth'], mode='lines', name='Khoảng tin cậy (dưới)', line=dict(color='yellow', width=0), fill='tonexty', fillcolor='rgba(255, 255, 0, 0.2)'))
    fig1.update_layout(title='Giá trị bán hàng hàng tháng - Prophet (Tập gốc)', xaxis_title='Ngày', yaxis_title='Giá trị bán hàng', xaxis_tickformat='%Y-%m', xaxis_tickangle=45, yaxis=dict(griddash='dash', gridcolor='gray'))
    st.plotly_chart(fig1)

    # Biểu đồ 2: Dự đoán 12 tháng tiếp theo
    future = model.make_future_dataframe(periods=365, freq='D')
    future_forecast = model.predict(future)
    future_forecast['yhat_smooth'] = future_forecast['yhat'].rolling(window=5, center=True, min_periods=1).mean()
    future_forecast['yhat_lower_smooth'] = future_forecast['yhat_lower'].rolling(window=5, center=True, min_periods=1).mean()
    future_forecast['yhat_upper_smooth'] = future_forecast['yhat_upper'].rolling(window=5, center=True, min_periods=1).mean()

    # Vẽ biểu đồ dự đoán tương lai
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ds_smooth, y=y_smooth, mode='lines', name='Thực tế', line=dict(color='blue', width=1)))
    fig2.add_trace(go.Scatter(x=future_forecast['ds'][future_forecast['ds'] <= prophet_df['ds'].max()],
                              y=future_forecast['yhat_smooth'][future_forecast['ds'] <= prophet_df['ds'].max()],
                              mode='lines', name='Dự đoán', line=dict(color='orange', width=1)))
    fig2.add_trace(go.Scatter(x=future_forecast['ds'][future_forecast['ds'] > prophet_df['ds'].max()],
                              y=future_forecast['yhat_smooth'][future_forecast['ds'] > prophet_df['ds'].max()],
                              mode='lines', name='Dự đoán tương lai', line=dict(color='red', width=1)))
    fig2.add_trace(go.Scatter(x=future_forecast['ds'][future_forecast['ds'] <= prophet_df['ds'].max()],
                              y=future_forecast['yhat_upper_smooth'][future_forecast['ds'] <= prophet_df['ds'].max()],
                              mode='lines', name='Khoảng tin cậy (trên)', line=dict(color='yellow', width=0), showlegend=False))
    fig2.add_trace(go.Scatter(x=future_forecast['ds'][future_forecast['ds'] <= prophet_df['ds'].max()],
                              y=future_forecast['yhat_lower_smooth'][future_forecast['ds'] <= prophet_df['ds'].max()],
                              mode='lines', name='Khoảng tin cậy (dưới)', line=dict(color='yellow', width=0), fill='tonexty', fillcolor='rgba(255, 255, 0, 0.2)'))
    fig2.add_trace(go.Scatter(x=future_forecast['ds'][future_forecast['ds'] > prophet_df['ds'].max()],
                              y=future_forecast['yhat_upper_smooth'][future_forecast['ds'] > prophet_df['ds'].max()],
                              mode='lines', name='Khoảng tin cậy tương lai (trên)', line=dict(color='pink', width=0), showlegend=False))
    fig2.add_trace(go.Scatter(x=future_forecast['ds'][future_forecast['ds'] > prophet_df['ds'].max()],
                              y=future_forecast['yhat_lower_smooth'][future_forecast['ds'] > prophet_df['ds'].max()],
                              mode='lines', name='Khoảng tin cậy tương lai (dưới)', line=dict(color='pink', width=0), fill='tonexty', fillcolor='rgba(255, 192, 203, 0.2)'))
    fig2.update_layout(title='Dự đoán giá trị bán hàng 12 tháng tiếp theo - Prophet', xaxis_title='Ngày', yaxis_title='Giá trị bán hàng', xaxis_tickformat='%Y-%m', xaxis_tickangle=45, yaxis=dict(griddash='dash', gridcolor='gray'))
    st.plotly_chart(fig2)

    # Hiển thị các chỉ số đánh giá
    eval_df = pd.merge(prophet_df[['ds', 'y']], past_forecast[['ds', 'yhat']], on='ds')
    mae = mean_absolute_error(eval_df['y'], eval_df['yhat'])
    rmse = np.sqrt(mean_squared_error(eval_df['y'], eval_df['yhat']))
    mape = np.mean(np.abs((eval_df['y'] - eval_df['yhat']) / eval_df['y'])) * 100
    r2 = r2_score(eval_df['y'], eval_df['yhat'])

    st.subheader("Đánh Giá Mô Hình Dự Đoán")
    st.write(f"📊 MAE: {mae:.2f}")
    st.write(f"📊 RMSE: {rmse:.2f}")
    st.write(f"📊 MAPE: {mape:.2f}%")
    st.write(f"📊 R² Score: {r2:.2f}")

# Tab 3: Phân Cụm Khách Hàng
with tab3:
    st.header("Phân Cụm Khách Hàng với GMM")

    # Lấy mẫu dữ liệu
    df_sample = df.sample(n=35000, random_state=42)
    df_cluster = df_sample[['Product Cost', 'Shipping Fee', 'Order Total', 'Profit']]

    # Chuẩn hóa và PCA
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cluster)
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)

    # Phân cụm với GMM
    gmm = GaussianMixture(n_components=7, random_state=42)
    clusters = gmm.fit_predict(df_pca)
    df_sample['Cluster'] = clusters

    # Vẽ biểu đồ phân cụm
    fig3 = px.scatter(x=df_pca[:, 0], y=df_pca[:, 1], color=clusters.astype(str), title='Phân Cụm Khách Hàng với GMM',
                      labels={'x': 'PCA 1', 'y': 'PCA 2', 'color': 'Cluster'}, color_discrete_sequence=px.colors.qualitative.T10)
    fig3.update_layout(showlegend=True)
    st.plotly_chart(fig3)

    # Hiển thị kết quả phân cụm
    st.subheader("Kết Quả Phân Cụm (Mẫu)")
    st.dataframe(df_sample[['Order Id', 'City', 'Country', 'Cluster']].head())

    # Phân tích đặc trưng từng cụm
    df_analysis = df_sample[['Product Cost', 'Shipping Fee', 'Order Total', 'Profit', 'Cluster']].copy()
    cluster_summary = df_analysis.groupby('Cluster').mean().round(2)
    cluster_counts = df_analysis['Cluster'].value_counts().sort_index()
    cluster_summary['Count'] = cluster_counts

    st.subheader("Đặc Trưng Trung Bình của Từng Cụm")
    st.dataframe(cluster_summary)

# Tab 4: Đánh Giá Mô Hình Phân Cụm
with tab4:
    st.header("Đánh Giá Mô Hình Phân Cụm")

    # Đánh giá phân cụm
    df_valid = df_sample.dropna(subset=['Cluster'])
    X_valid = df_pca
    labels = df_valid['Cluster']

    sil_score = silhouette_score(X_valid, labels)
    db_index = davies_bouldin_score(X_valid, labels)
    ch_index = calinski_harabasz_score(X_valid, labels)

    st.write(f"📊 Silhouette Score: {sil_score:.3f}")
    st.write(f"📊 Davies-Bouldin Index: {db_index:.3f}")
    st.write(f"📊 Calinski-Harabasz Index: {ch_index:.3f}")
    st.write(f"📊 Số cụm: {len(set(labels))}")

# Footer
st.markdown("---")
st.markdown("Ứng dụng được xây dựng với Streamlit bởi xAI. Liên hệ hỗ trợ: support@x.ai")