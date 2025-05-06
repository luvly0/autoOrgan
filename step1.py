def generate_index_and_metadata():

    import os
    import pandas as pd
    import faiss
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    import numpy as np
    
    # 임베딩, 클러스터링, 인덱싱 파일 생성
    # 업로드된 파일 경로
    UPLOAD_FOLDER = './uploads'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # 결과 파일을 저장할 폴더
    OUTPUT_FOLDER = './outputs'
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 모델 로딩
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 임베딩 및 클러스터링 결과 저장 위치
    file_metadata_path = os.path.join(OUTPUT_FOLDER, 'file_metadata.csv')
    file_cluster_labels_path = os.path.join(OUTPUT_FOLDER, 'file_cluster_labels.txt')
    file_index_path = os.path.join(OUTPUT_FOLDER, 'file_index.faiss')
    
    # 예시로 사용할 파일 목록 (업로드된 파일을 활용하도록 수정 가능)
    documents = [
        "회의록 내용",
        "2025 상반기 대출실적 내용",
        "앱개발기획서 내용",
        "고객연락처 정리 내용",
        "투자보고서 2025Q1 내용"
    ]
    
    # 임베딩 생성
    embeddings = model.encode(documents).astype("float32")
    
    # 클러스터링 (예시: 3개 클러스터로)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # FAISS 인덱스 생성
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    # FAISS 인덱스 저장
    faiss.write_index(index, file_index_path)
    
    # 클러스터 레이블 저장
    np.savetxt(file_cluster_labels_path, clusters, fmt="%d")
    
    # 메타데이터 저장
    metadata = pd.DataFrame({
        "index_id": range(len(documents)),
        "file_name": ["회의록.txt", "2025_상반기_대출실적.txt", "앱개발기획서.txt", "고객연락처_정리.xlsx", "투자보고서_2025Q1.pdf"],
        "file_path": ["./docs/회의록.txt", "./docs/2025_상반기_대출실적.txt", "./docs/앱개발기획서.txt", "./docs/고객연락처_정리.xlsx", "./docs/투자보고서_2025Q1.pdf"],
        "cluster_name": ["회의", "대출실적", "기획서", "고객정보", "투자보고서"]
    })
    
    metadata.to_csv(file_metadata_path, index=False)
    
    print("✅ 인덱스, 메타데이터, 클러스터 파일 생성 완료.")


