import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st

def search_files(query):
    # FAISS 인덱스 및 클러스터링, 메타데이터 파일 경로
    UPLOAD_FOLDER = './uploads'
    OUTPUT_FOLDER = './outputs'
    file_metadata_path = os.path.join(OUTPUT_FOLDER, 'file_metadata.csv')
    file_cluster_labels_path = os.path.join(OUTPUT_FOLDER, 'file_cluster_labels.txt')
    file_index_path = os.path.join(OUTPUT_FOLDER, 'file_index.faiss')

    # 파일이 존재하는지 확인
    if not os.path.exists(file_metadata_path) or not os.path.exists(file_index_path) or not os.path.exists(file_cluster_labels_path):
        st.error("파일이 존재하지 않습니다. 먼저 인덱스를 생성해야 합니다.")
        return

    # 메타데이터 로드
    metadata = pd.read_csv(file_metadata_path)

    # FAISS 인덱스 로드
    index = faiss.read_index(file_index_path)

    # 클러스터 레이블 로드
    clusters = np.loadtxt(file_cluster_labels_path, dtype=int)

    # 모델 로딩
    model = SentenceTransformer("all-MiniLM-L6-v2")

    if query:
        # 검색어 임베딩
        query_vec = model.encode([query]).astype("float32")

        # 유사도 검색
        D, I = index.search(query_vec, k=5)  # 가장 유사한 5개 파일 찾기
        
        # 유사한 파일 결과를 리스트로 저장
        result_df = []
        for idx, dist in zip(I[0], D[0]):
            fname = metadata.loc[metadata["index_id"] == idx, "file_name"].values[0]
            cluster_name = metadata.loc[metadata["index_id"] == idx, "cluster_name"].values[0]
            similarity = 1 - dist  # 유사도 계산 (1 - 거리)
            result_df.append({"file_name": fname, "similarity": similarity, "cluster_name": cluster_name})

        # DataFrame으로 변환
        result_df = pd.DataFrame(result_df)


        # 유사도 순으로 정렬 및 출력
        for cluster in result_df['cluster_name'].unique():
            cluster_results = result_df[result_df['cluster_name'] == cluster].sort_values(by="similarity", ascending=False)
            
            for _, row in cluster_results.iterrows():
                st.write(f"{cluster} / {row['file_name']} (유사도: {row['similarity']*100:.2f}%)")
                
