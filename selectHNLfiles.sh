cd /home/hep/jd918/project/LLP/nanoAOD_files_200417/2016/
path="/home/hep/jd918/project/LLP/nanoAOD_files_200417/2016/"
files_hnl=$(ls | grep "HeavyNeutrino")
echo $files_hnl
files_cc=$(ls | grep "HeavyNeutrino" | grep "_cc")
echo $files_cc
files_nocc=$(echo "$files_cc" | sed 's/_cc//g')
echo $files_nocc
files_no_cc=$(comm -12 <(echo "$files_nocc") <(echo "$files_hnl"))
echo $files_no_cc
while IFS=" " read -r LINE
do
  echo "${path}${LINE}"
done <<< "$files_no_cc"

qsub -t 1-$(./get_job_number.sh) job-runHNL.sh
