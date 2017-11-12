mv ./data/augdata/IMG/* ./data/IMG/
echo 'Copy imgs finished'
cat ./data/augdata/driving_log.csv  >> ./data/driving_log.csv
echo 'driving_log.csv has been appended!'
rm -rf ./data/augdata
rm -rf ./data/augdata.zip
echo '.data/augdata and augdata.zip folders are deleted!'