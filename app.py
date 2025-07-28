import streamlit as st
import pandas as pd
import plotly.express as px
from Bio.PDB import PDBParser, Polypeptide
import os
import io
import py3Dmol
from stpyvmol import st_pyvmol

# --- UI 설정 ---
st.set_page_config(layout="wide")
st.title("🔬 CAR-SAR: AI 바인더 분석 시스템 (v1.5)")
st.write("---")

# --- 데모 데이터 생성 함수 ---
def create_dummy_data_if_needed():
    """데모용 CSV와 PDB 파일이 없으면 생성합니다."""
    # Demo CSV
    demo_csv_path = "data/binders_demo.csv"
    if not os.path.exists(demo_csv_path):
        csv_data = """Binder_ID,Sequence (CDR3),Target_Antigen,KD (nM)
BKR-01,ARDYFGYGMDVW,BCMA,10.5
BKR-02,ARDYFWYGMDVW,BCMA,0.1
BKR-03,VRSKMDSSYFDY,BCMA,8.7
BKR-04,TRGSSYVLDAM,BCMA,120.3
BKR-05,GYDFWSGAYDY,BCMA,55.4
BKR-06,GYDFWSGAYEY,BCMA,2.1
"""
        with open(demo_csv_path, "w") as f:
            f.write(csv_data)

    # Demo PDBs
    pdb_dir = "data/pdb_files_demo"
    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir)

    pdb_content_template = """
ATOM      1  N   ALA A   1      27.340  16.433  27.945  1.00  0.00           N
ATOM      2  CA  ALA A   1      26.266  15.548  27.433  1.00  0.00           C
"""
    binder_ids = ["BKR-01", "BKR-02", "BKR-03", "BKR-04", "BKR-05", "BKR-06"]
    for binder_id in binder_ids:
        pdb_path = os.path.join(pdb_dir, f"{binder_id}.pdb")
        if not os.path.exists(pdb_path):
            with open(pdb_path, "w") as f:
                f.write(pdb_content_template)
    return demo_csv_path, pdb_dir

# --- 사이드바: 데이터 업로드 기능 ---
st.sidebar.title("📁 데이터 소스 선택")
default_csv_path, default_pdb_folder = create_dummy_data_if_needed()

csv_path = st.sidebar.text_input("1. 바인더 데이터 (CSV) 경로", value=default_csv_path)
pdb_folder_path = st.sidebar.text_input("2. PDB 파일 폴더 경로", value=default_pdb_folder)


# --- 데이터 로드 로직 ---
df = None
try:
    df = pd.read_csv(csv_path)
    st.sidebar.success("데이터 로드 성공!")
except FileNotFoundError:
    st.sidebar.error(f"오류: '{csv_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
except Exception as e:
    st.sidebar.error(f"데이터 로드 오류: {e}")

if df is not None:
    # (이하 분석 및 UI 코드는 이전과 동일하게 유지)
    def get_structure_info(pdb_path, binder_seq_from_csv=""):
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("binder", pdb_path)
            model = structure[0]
            chains = [chain.id for chain in model]
            residue_count = sum(len(list(chain.get_residues())) for chain in model)
            pdb_seq = "".join(Polypeptide.three_to_one(res.get_resname()) for chain in model for res in chain if Polypeptide.is_aa(res))
            seq_match = "✅ 일치" if binder_seq_from_csv in pdb_seq else "❌ 불일치" if binder_seq_from_csv else "N/A"
            return {"Chains": chains, "Residue Count": residue_count, "Sequence Match": seq_match, "Status": "Success"}
        except Exception as e:
            return {"Status": "Error", "Message": str(e)}

    st.header("1. 바인더 데이터셋 개요")
    st.dataframe(df)

    st.header("2. 기초 데이터 탐색 (EDA)")
    if 'KD (nM)' in df.columns:
        df['pKD'] = -pd.np.log10(df['KD (nM)'] * 1e-9)
        fig = px.histogram(df, x="pKD", title="<b>결합 친화도 분포 (pKD)</b>", labels={'pKD':'pKD (-log10(KD))'}, nbins=20)
        st.plotly_chart(fig, use_container_width=True)

    st.header("3. 🧬 단백질 구조 분석 및 3D 시각화")
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_binder = st.selectbox("분석할 바인더 선택:", df['Binder_ID'])
        if selected_binder:
            pdb_path = os.path.join(pdb_folder_path, f"{selected_binder}.pdb")
            st.info(f"**분석 대상: {selected_binder}**")
            info = get_structure_info(pdb_path, df[df['Binder_ID'] == selected_binder].get('Sequence (CDR3)', pd.Series([""])).iloc[0])
            if info["Status"] == "Success":
                st.metric("총 잔기 (Residues)", info["Residue Count"])
                st.metric("체인 (Chains)", ", ".join(info["Chains"]))
                st.metric("CSV-PDB 서열 비교", info["Sequence Match"])
            else:
                st.error(f"구조 분석 오류: {info['Message']}")
    with col2:
        if selected_binder:
            pdb_path = os.path.join(pdb_folder_path, f"{selected_binder}.pdb")
            if os.path.exists(pdb_path):
                st.subheader("3D 구조 뷰어")
                view = py3Dmol.view(width=600, height=400)
                view.addModel(open(pdb_path, 'r').read(), 'pdb')
                view.setStyle({'cartoon': {'color': 'spectrum'}})
                view.zoomTo()
                st_pyvmol(view)
