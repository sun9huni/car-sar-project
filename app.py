import streamlit as st
import pandas as pd
import plotly.express as px
from Bio.PDB import PDBParser, Polypeptide
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import os
import itertools
import py3Dmol
from stpyvmol import st_pyvmol

# --- Levenshtein Distance (편집 거리) 함수 ---
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

# --- UI 설정 ---
st.set_page_config(layout="wide")
st.title("🔬 CAR-SAR: AI 바인더 분석 시스템 (v2.0)")
st.subheader("Phase 2: Affinity Cliff 분석")
st.write("---")

# --- 데모 데이터 생성 함수 (v1.5와 동일) ---
def create_dummy_data_if_needed():
    """데모용 CSV와 PDB 파일이 없으면 생성합니다."""
    # Demo CSV
    demo_csv_path = "data/binders_demo.csv"
    if not os.path.exists("data"): os.makedirs("data")
    if not os.path.exists(demo_csv_path):
        csv_data = """Binder_ID,Sequence (CDR3),Target_Antigen,KD (nM)
BKR-01,ARDYFGYGMDVW,BCMA,10.5
BKR-02,ARDYFWYGMDVW,BCMA,0.1
BKR-03,VRSKMDSSYFDY,BCMA,8.7
BKR-04,TRGSSYVLDAM,BCMA,120.3
BKR-05,GYDFWSGAYDY,BCMA,55.4
BKR-06,GYDFWSGAYEY,BCMA,2.1
BKR-07,ARDYFAYGMDVW,BCMA,9.8
"""
        with open(demo_csv_path, "w") as f: f.write(csv_data)
    
    pdb_dir = "data/pdb_files_demo"
    if not os.path.exists(pdb_dir): os.makedirs(pdb_dir)
    pdb_content_template = "ATOM      1  N   ALA A   1      27.340  16.433  27.945  1.00  0.00           N\nATOM      2  CA  ALA A   1      26.266  15.548  27.433  1.00  0.00           C\n"
    binder_ids = ["BKR-01", "BKR-02", "BKR-03", "BKR-04", "BKR-05", "BKR-06", "BKR-07"]
    for binder_id in binder_ids:
        pdb_path = os.path.join(pdb_dir, f"{binder_id}.pdb")
        if not os.path.exists(pdb_path):
            with open(pdb_path, "w") as f: f.write(pdb_content_template)
    return demo_csv_path, pdb_dir

# --- 사이드바 ---
st.sidebar.title("📁 데이터 소스")
default_csv_path, default_pdb_folder = create_dummy_data_if_needed()
csv_path = st.sidebar.text_input("1. 바인더 데이터 (CSV) 경로", value=default_csv_path)
pdb_folder_path = st.sidebar.text_input("2. PDB 파일 폴더 경로", value=default_pdb_folder)

# --- 데이터 로드 ---
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    st.error(f"오류: '{csv_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    st.stop()

# --- Phase 2: Affinity Cliff 분석 섹션 ---
st.header("1. Affinity Cliff 탐지")
st.write("구조는 유사하지만 활성 차이가 큰 바인더 쌍을 자동으로 탐지합니다.")

# 탐지 파라미터 설정
col1, col2 = st.columns(2)
seq_diff_threshold = col1.number_input("서열 차이 (최대 편집 거리)", min_value=1, max_value=10, value=2, help="두 서열 간의 편집 거리(Levenshtein distance)가 이 값 이하인 쌍을 유사하다고 판단합니다.")
kd_fold_threshold = col2.number_input("KD값 변화 (최소 배수)", min_value=2.0, max_value=1000.0, value=50.0, step=10.0, help="두 바인더의 KD값 차이가 이 배수 이상인 경우를 유의미한 활성 변화로 판단합니다.")

if st.button("📈 Affinity Cliff 탐지 시작"):
    cliff_pairs = []
    # 데이터프레임의 모든 행 쌍에 대해 반복
    for (idx1, row1), (idx2, row2) in itertools.combinations(df.iterrows(), 2):
        seq1 = row1['Sequence (CDR3)']
        seq2 = row2['Sequence (CDR3)']
        
        # 1. 서열 유사도 체크
        dist = levenshtein_distance(seq1, seq2)
        if dist <= seq_diff_threshold:
            # 2. KD 값 변화 체크
            kd1 = row1['KD (nM)']
            kd2 = row2['KD (nM)']
            if kd1 > 0 and kd2 > 0:
                fold_change = kd1 / kd2 if kd1 > kd2 else kd2 / kd1
                if fold_change >= kd_fold_threshold:
                    # 더 나은 바인더(KD 낮은 쪽)를 기준으로 저장
                    better_binder, worse_binder = (row2, row1) if kd1 > kd2 else (row1, row2)
                    cliff_pairs.append({
                        "Pair": f"{worse_binder['Binder_ID']} vs {better_binder['Binder_ID']}",
                        "Worse_Binder": worse_binder['Binder_ID'],
                        "Worse_KD": worse_binder['KD (nM)'],
                        "Better_Binder": better_binder['Binder_ID'],
                        "Better_KD": better_binder['KD (nM)'],
                        "Fold_Improvement": fold_change,
                        "Sequence_Distance": dist
                    })

    if cliff_pairs:
        st.success(f"총 {len(cliff_pairs)}개의 Affinity Cliff 쌍을 찾았습니다.")
        cliff_df = pd.DataFrame(cliff_pairs)
        st.dataframe(cliff_df.style.format({"Fold_Improvement": "{:.1f}x"}))
        st.session_state['cliff_df'] = cliff_df # 세션에 저장
    else:
        st.warning("설정된 기준으로 Affinity Cliff 쌍을 찾지 못했습니다. 기준을 조정해보세요.")

st.write("---")

# --- Phase 2: 구조 비교 분석 ---
st.header("2. 3D 구조 비교 분석")
st.write("탐지된 Affinity Cliff 쌍의 3D 구조를 중첩하여 차이점을 시각적으로 확인합니다.")

if 'cliff_df' in st.session_state:
    cliff_df = st.session_state['cliff_df']
    
    # 비교할 쌍 선택
    selected_pair_id = st.selectbox("분석할 Affinity Cliff 쌍을 선택하세요:", cliff_df['Pair'])
    
    if selected_pair_id:
        selected_pair_data = cliff_df[cliff_df['Pair'] == selected_pair_id].iloc[0]
        
        binder1_id = selected_pair_data['Worse_Binder']
        binder2_id = selected_pair_data['Better_Binder']
        
        binder1_pdb = os.path.join(pdb_folder_path, f"{binder1_id}.pdb")
        binder2_pdb = os.path.join(pdb_folder_path, f"{binder2_id}.pdb")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**기준 구조 (Worse): {binder1_id}**")
            st.metric("KD (nM)", selected_pair_data['Worse_KD'])
            if os.path.exists(binder1_pdb):
                with open(binder1_pdb, 'r') as f:
                    st.text_area("PDB Content", f.read(), height=150)
            else:
                st.error("PDB 파일 없음")
        
        with col2:
            st.info(f"**비교 구조 (Better): {binder2_id}**")
            st.metric("KD (nM)", selected_pair_data['Better_KD'])
            if os.path.exists(binder2_pdb):
                with open(binder2_pdb, 'r') as f:
                    st.text_area("PDB Content", f.read(), height=150)
            else:
                st.error("PDB 파일 없음")
        
        st.subheader("3D 구조 중첩(Superimpose) 비교")
        
        if os.path.exists(binder1_pdb) and os.path.exists(binder2_pdb):
            # Py3Dmol 뷰어 생성
            view = py3Dmol.view(width=800, height=600)
            
            # 모델 로드
            view.addModel(open(binder1_pdb, 'r').read(), 'pdb', {'model': 0})
            view.addModel(open(binder2_pdb, 'r').read(), 'pdb', {'model': 1})
            
            # 구조 중첩 (model 1을 model 0에 맞춤)
            view.superpose({'model': 1}, {'model': 0})
            
            # 스타일링
            view.setStyle({'model': 0}, {'cartoon': {'color': 'blue'}}) # Worse: Blue
            view.setStyle({'model': 1}, {'cartoon': {'color': 'red'}}) # Better: Red
            
            view.zoomTo()
            st_pyvmol(view)
        else:
            st.error("두 PDB 파일이 모두 있어야 구조를 비교할 수 있습니다.")
else:
    st.info("'Affinity Cliff 탐지 시작' 버튼을 눌러 분석을 먼저 실행해주세요.")

