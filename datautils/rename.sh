TOT=$(ls ./dataset/* | wc -l)
CTR=0
for f in $(find ./dataset/ -type f); do
	CTR=$((CTR+1))
	echo $CTR/$TOT
	if $(echo $f | grep "+" > /dev/null); then
		newf=$(echo $f | sed 's/\-2/\-Op\-padding/g')
	else
		newf=$f-Ob-padding
	fi
	mv $f $newf
done
