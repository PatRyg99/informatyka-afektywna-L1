#echo "DGCNN"
#python -m run.inference -i output/dgcnn-xyz -f xyz
#python -m run.inference -i output/dgcnn-hks -f hks
#python -m run.inference -i output/dgcnn-xyz-rot -f xyz
#python -m run.inference -i output/dgcnn-hks-rot -f hks

#echo "SAGE"
#python -m run.inference -i output/sage-xyz -f xyz
#python -m run.inference -i output/sage-hks -f hks
#python -m run.inference -i output/sage-xyz-rot -f xyz
#python -m run.inference -i output/sage-hks-rot -f hks

echo "FEAST"
python -m run.inference -i output/feast-xyz -f xyz
python -m run.inference -i output/feast-hks -f hks
python -m run.inference -i output/feast-xyz-rot -f xyz
python -m run.inference -i output/feast-hks-rot -f hks
