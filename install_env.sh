
#python = 3.10, NO GLIBC_2.29
conda install -y pip

#stl
conda install -y pandas
conda install -y matplotlib
conda install -y seaborn
conda install -y tqdm
conda install -y ipykernel
conda install -y jupyter

#data processing
conda install -y scikit-learn
conda install -y scipy

#kernel
python -m ipykernel install --user --name="pyg_CUDA"

conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install torch_geometric

# Install specific PyG versions
pip install \
    pyg_lib==0.3.1+pt21cu121 \
    torch_scatter==2.1.2+pt21cu121 \
    torch_sparse==0.6.18+pt21cu121 \
    torch_cluster==1.6.3+pt21cu121 \
    torch_spline_conv==1.2.2+pt21cu121 \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html


pip install missingno networkx "numpy<2.0"


