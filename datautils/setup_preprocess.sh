./clean.sh
cp ../../../Binaries/Dataset-Muaz/* dataset/

echo "Running run.py..."
LD_LIBRARY_PATH=/home/gab/anaconda3/envs/jtrans/lib/  python3 run.py
echo "Done, check ./extract"

mkdir -p extract/Dataset-Muaz/
mv extract/*.pkl extract/Dataset-Muaz/
