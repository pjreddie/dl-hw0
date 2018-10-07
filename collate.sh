rm -rf submit/
mkdir -p submit

prepare () {
    if [[ $(git diff $1 | wc -c) -eq 0 ]]; then 
        echo "WARNING: $1 is unchanged according to git."
    fi
    cp $1 submit/
}

echo "Creating tarball..."
prepare src/matrix.c
prepare src/activations.c
prepare src/connected_layer.c
prepare mnist_pytorch.ipynb

tar cvzf submit.tar.gz submit
rm -rf submit/
echo "Done. Please upload submit.tar.gz to Canvas."

