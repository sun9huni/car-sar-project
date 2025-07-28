import streamlit as st
import pandas as pd
import os
import itertools
import py3Dmol
import requests
import numpy as np
import plotly.graph_objects as go
from Bio.PDB import PDBList, PDBParser, Superimposer
from Bio.SeqUtils import seq1
from requests.exceptions import RequestException
import streamlit.components.v1 as components

# FutureHouseClient is optional if API key is not provided
try:
    from futurehouse_client import FutureHouseClient, JobNames
    FUTUREHOUSE_AVAILABLE = True
except ImportError:
    FUTUREHOUSE_AVAILABLE = False

# --- Helper Functions ---

def levenshtein_distance(s1, s2):
    """Calculates the Levenshtein distance between two sequences."""
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

@st.cache_data
def get_pdb_structure(pdb_id):
    """Fetches PDB data from RCSB and returns a parsed Structure object."""
    pdb_dir = "data/pdb_cache"
    os.makedirs(pdb_dir, exist_ok=True)
    pdbl = PDBList()
    try:
        # Retrieve PDB file, the actual filename can have a different case
        pdb_path_generic = pdbl.retrieve_pdb_file(pdb_id, pdir=pdb_dir, file_format='pdb', overwrite=False)
        # Construct the expected path for Bio.PDB
        pdb_path_specific = os.path.join(pdb_dir, f"pdb{pdb_id.lower()}.ent")

        if not os.path.exists(pdb_path_specific):
             st.error(f"PDB 파일을 다운로드하지 못했습니다: {pdb_id}")
             return None
        parser = PDBParser(QUIET=True)
        return parser.get_structure(pdb_id, pdb_path_specific)
    except Exception as e:
        st.error(f"PDB 파일 처리 중 오류 발생: {pdb_id}, {e}")
        return None


def get_chain_sequence(chain_obj):
    """Extracts 1-letter amino acid sequence from a PDB Chain object."""
    return seq1("".join([res.get_resname() for res in chain_obj if res.id[0] == ' ']))

def get_mutations(residue_list1, residue_list2):
    """Compares two residue lists and returns mutation info."""
    mutations = []
    min_len = min(len(residue_list1), len(residue_list2))
    for i in range(min_len):
        if residue_list1[i].get_resname() != residue_list2[i].get_resname():
             mutations.append({
                "position_worse": residue_list1[i].id[1],
                "from": seq1(residue_list1[i].get_resname()),
                "position_better": residue_list2[i].id[1],
                "to": seq1(residue_list2[i].get_resname())
            })
    return mutations


def calculate_rmsd(chain1, chain2):
    """Superimposes two Chain structures and calculates the RMSD."""
    super_imposer = Superimposer()
    atoms1 = [atom for atom in chain1.get_atoms() if atom.get_name() == 'CA']
    atoms2 = [atom for atom in chain2.get_atoms() if atom.get_name() == 'CA']

    if len(atoms1) != len(atoms2):
        min_len = min(len(atoms1), len(atoms2))
        atoms1, atoms2 = atoms1[:min_len], atoms2[:min_len]

    if not atoms1: return None, "Error: No C-alpha atoms in Chain 1."

    super_imposer.set_atoms(atoms1, atoms2)
    return super_imposer.rms, "Success"

def generate_llm_prompt(pair_data, rmsd, mutations):
    """Generates an LLM prompt based on the analysis results."""
    mutation_str = ", ".join(f"{m['from']}{m['position_worse']}->{m['to']}{m['position_better']}" for m in mutations) if mutations else "서열 동일"
    prompt = f"""
### **분석 보고: Affinity Cliff 원인 분석 요청**
**1. 분석 대상:**
- **기준 바인더 (Worse):** {pair_data['Worse_Binder']} (Chain: {pair_data['Worse_Chain']}, KD: {pair_data['Worse_KD']:.2f} nM)
- **개선 바인더 (Better):** {pair_data['Better_Binder']} (Chain: {pair_data['Better_Chain']}, KD: {pair_data['Better_KD']:.2f} nM)
**2. 핵심 변화 요약:**
- **활성도 변화:** 결합 친화도가 **{pair_data['Fold_Improvement']:.1f}배** 향상됨.
- **구조적 변이:** {mutation_str}.
- **구조적 유사도 (RMSD):** 두 바인더 Chain의 C-alpha 원자 기준 RMSD는 **{rmsd:.3f} Å**으로 매우 유사함.
**3. 분석 요청:**
위 정보를 바탕으로, 두 바인더의 구조적 차이가 어떻게 결합 친화도를 {pair_data['Fold_Improvement']:.1f}배나 극적으로 향상시켰는지에 대한 **구조적, 물리화학적 가설 3가지를 제시해줘.** 각 가설은 PDB 구조 내에서 예상되는 **원자 간 상호작용(수소결합, 염다리, 소수성 상호작용 등)의 변화**를 구체적으로 언급해야 하며, 해당 가설을 검증하기 위한 **다음 실험 단계를 제안해줘.**
"""
    return prompt.strip()

def create_aligned_side_by_side_view(pdb_id_1, chain_1, pdb_id_2, chain_2, highlight_muts=False, mutations=None):
    """Creates a linked, side-by-side 3D view and returns it as an HTML string."""
    try:
        pdb_url_1, pdb_url_2 = f"https://files.rcsb.org/view/{pdb_id_1}.pdb", f"https://files.rcsb.org/view/{pdb_id_2}.pdb"
        pdb_data_1, pdb_data_2 = requests.get(pdb_url_1).text, requests.get(pdb_url_2).text

        # Use linked viewers for synchronized camera movements
        view = py3Dmol.view(width=1200, height=500, linked=True, viewergrid=(1,2))
        
        # Add and style the first model
        view.addModel(pdb_data_1, 'pdb', viewer=(0,0))
        view.setStyle({'chain': chain_1}, {'cartoon': {'color': 'blue'}}, viewer=(0,0))
        view.zoomTo({'chain': chain_1}, viewer=(0,0))
        
        # Add and style the second model
        view.addModel(pdb_data_2, 'pdb', viewer=(0,1))
        view.setStyle({'chain': chain_2}, {'cartoon': {'color': 'red'}}, viewer=(0,1))
        view.zoomTo({'chain': chain_2}, viewer=(0,1))
        
        if highlight_muts and mutations:
            resi_list_1 = [m['position_worse'] for m in mutations]
            resi_list_2 = [m['position_better'] for m in mutations]
            view.addStyle({'chain': chain_1, 'resi': resi_list_1}, {'stick': {'colorscheme': 'blueCarbon', 'radius': 0.2}}, viewer=(0,0))
            view.addStyle({'chain': chain_2, 'resi': resi_list_2}, {'stick': {'colorscheme': 'redCarbon', 'radius': 0.2}}, viewer=(0,1))

        return view._make_html()
    except Exception as e:
        return f"<p>HTML 뷰어 생성 중 오류 발생: {e}</p>"


def create_distance_map(chain1, chain2):
    """Creates a difference distance map between two chains."""
    atoms1 = [atom.get_coord() for atom in chain1.get_atoms() if atom.get_name() == 'CA']
    atoms2 = [atom.get_coord() for atom in chain2.get_atoms() if atom.get_name() == 'CA']

    min_len = min(len(atoms1), len(atoms2))
    dist_matrix1 = np.linalg.norm(np.array(atoms1[:min_len])[:, np.newaxis, :] - np.array(atoms1[:min_len])[np.newaxis, :, :], axis=2)
    dist_matrix2 = np.linalg.norm(np.array(atoms2[:min_len])[:, np.newaxis, :] - np.array(atoms2[:min_len])[np.newaxis, :, :], axis=2)

    diff_matrix = np.abs(dist_matrix1 - dist_matrix2)

    fig = go.Figure(data=go.Heatmap(z=diff_matrix, colorscale='Reds'))
    fig.update_layout(title='잔기 간 거리 차이 맵 (붉을수록 구조 변화가 큰 지역)', xaxis_title="잔기 번호", yaxis_title="잔기 번호")
    return fig

# --- UI Layout ---
st.set_page_config(layout="wide")
st.title("🔬 CAR-SAR: AI 바인더 분석 시스템 (v17.0 - 3D 정렬 비교)")
st.write("---")

# Sidebar
st.sidebar.title("📁 데이터")
DEMO_DATA = {
    "Binder_ID": ["7L7D", "7L7E", "7N3G", "7N3H"],
    "Chain_ID": ['H', 'H', 'H', 'H'],
    "Sequence_for_dist": ["QYSTVPWTF", "QYSTVPWAF", "GYCSGGFSCYV", "GYCSGGFSCYW"],
    "KD (nM)": [15.0, 0.8, 25.0, 1.2]
}
df = pd.DataFrame(DEMO_DATA)
st.sidebar.dataframe(df)

st.sidebar.title("🔑 API 설정")
api_key = st.sidebar.text_input("FutureHouse API Key", type="password")

# Main content
st.header("1. Affinity Cliff 탐지")
col1, col2 = st.columns(2)
seq_dist_threshold = col1.number_input("서열 차이 (최대 편집 거리)", 1, 10, 2)
kd_fold_threshold = col2.number_input("KD값 변화 (최소 배수)", 2.0, 100.0, 10.0, step=5.0)

cliff_pairs = []
for (idx1, row1), (idx2, row2) in itertools.combinations(df.iterrows(), 2):
    dist = levenshtein_distance(row1['Sequence_for_dist'], row2['Sequence_for_dist'])
    if dist <= seq_dist_threshold:
        kd1, kd2 = row1['KD (nM)'], row2['KD (nM)']
        if kd1 > 0 and kd2 > 0:
            fold_change = kd1 / kd2 if kd1 > kd2 else kd2 / kd1
            if fold_change >= kd_fold_threshold:
                better, worse = (row2, row1) if kd1 > kd2 else (row1, row2)
                cliff_pairs.append({
                    "Pair": f"{worse['Binder_ID']} vs {better['Binder_ID']}",
                    "Worse_Binder": worse['Binder_ID'], "Worse_Chain": worse['Chain_ID'], "Worse_KD": worse['KD (nM)'],
                    "Better_Binder": better['Binder_ID'], "Better_Chain": better['Chain_ID'], "Better_KD": better['KD (nM)'],
                    "Fold_Improvement": fold_change
                })

if cliff_pairs:
    st.success(f"총 {len(cliff_pairs)}개의 Affinity Cliff 쌍을 찾았습니다.")
    cliff_df = pd.DataFrame(cliff_pairs)
    st.dataframe(cliff_df[['Pair', 'Worse_KD', 'Better_KD', 'Fold_Improvement']].style.format({"Fold_Improvement": "{:.1f}x"}))
    st.session_state['cliff_df'] = cliff_df
else:
    st.warning("설정된 기준으로 Affinity Cliff 쌍을 찾지 못했습니다.")
    st.stop()

st.write("---")
st.header("2. 🎯 심층 분석 및 AI 가설 생성")
cliff_df = st.session_state.get('cliff_df')
if cliff_df is not None:
    selected_pair_id = st.selectbox("분석할 쌍을 선택하세요:", cliff_df['Pair'])

    if selected_pair_id:
        pair_data = cliff_df[cliff_df['Pair'] == selected_pair_id].iloc[0]
        worse_id, worse_chain_id = pair_data['Worse_Binder'], pair_data['Worse_Chain']
        better_id, better_chain_id = pair_data['Better_Binder'], pair_data['Better_Chain']

        try:
            struct_w = get_pdb_structure(worse_id)
            struct_b = get_pdb_structure(better_id)

            if struct_w and struct_b and worse_chain_id in struct_w[0] and better_chain_id in struct_b[0]:
                chain_w = struct_w[0][worse_chain_id]
                chain_b = struct_b[0][better_chain_id]

                # Use residue objects for mutation detection
                residues_w = [res for res in chain_w if res.id[0] == ' ']
                residues_b = [res for res in chain_b if res.id[0] == ' ']
                mutations = get_mutations(residues_w, residues_b)
                
                rmsd_val, rmsd_stat = calculate_rmsd(chain_w, chain_b)

                c1, c2 = st.columns([1, 1])
                with c1:
                    st.subheader("분석 요약")
                    st.metric("구조적 유사도 (RMSD)", f"{rmsd_val:.3f} Å" if rmsd_val is not None else "계산 불가")
                    if rmsd_stat != "Success": st.warning(rmsd_stat)
                    st.text("실제 서열 변이"); st.table(pd.DataFrame(mutations))

                with c2:
                    st.subheader("✍️ AI 가설 생성")
                    prompt = generate_llm_prompt(pair_data, rmsd_val, mutations)
                    st.text_area("LLM 프롬프트", prompt, height=265, key="prompt")
                    if st.button("🤖 AI로 가설 생성하기"):
                        st.success("AI 가설 생성 완료 (데모)")

                st.subheader("3. 🧬 구조 시각화 분석")

                vis_method = st.radio("시각화 방법 선택:",
                                      ["정렬하여 나란히 비교하기", "내부 거리 차이 맵"],
                                      horizontal=True)

                if vis_method == "정렬하여 나란히 비교하기":
                    highlight = st.checkbox("변이 아미노산 강조하기")
                    html_content = create_aligned_side_by_side_view(worse_id, worse_chain_id, better_id, better_chain_id, highlight_muts=highlight, mutations=mutations)
                    components.html(html_content, height=520)

                elif vis_method == "내부 거리 차이 맵":
                    fig = create_distance_map(chain_w, chain_b)
                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.error(f"PDB 구조 또는 지정된 체인을 찾지 못했습니다. PDB ID: {worse_id}({worse_chain_id}), {better_id}({better_chain_id})")
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")
else:
    st.info("먼저 Affinity Cliff를 탐지해주세요.")




