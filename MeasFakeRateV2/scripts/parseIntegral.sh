#!/bin/sh
ERA=$1
export PATH="${PWD}/python:${PATH}"
rm output/$ERA/*/fitresult.*.csv

parseIntegral.py --era $ERA --hlt MeasFakeEl12 --wp loose --syst Central >> output/$ERA/MeasFakeEl12/fitresult.loose.Central.csv
parseIntegral.py --era $ERA --hlt MeasFakeEl12 --wp tight --syst Central >> output/$ERA/MeasFakeEl12/fitresult.tight.Central.csv
parseIntegral.py --era $ERA --hlt MeasFakeEl23 --wp loose --syst Central >> output/$ERA/MeasFakeEl23/fitresult.loose.Central.csv
parseIntegral.py --era $ERA --hlt MeasFakeEl23 --wp tight --syst Central >> output/$ERA/MeasFakeEl23/fitresult.tight.Central.csv
parseIntegral.py --era $ERA --hlt MeasFakeEl12 --wp loose --syst RequireHeavyTag >> output/$ERA/MeasFakeEl12/fitresult.loose.RequireHeavyTag.csv
parseIntegral.py --era $ERA --hlt MeasFakeEl12 --wp tight --syst RequireHeavyTag >> output/$ERA/MeasFakeEl12/fitresult.tight.RequireHeavyTag.csv
parseIntegral.py --era $ERA --hlt MeasFakeEl23 --wp loose --syst RequireHeavyTag >> output/$ERA/MeasFakeEl23/fitresult.loose.RequireHeavyTag.csv
parseIntegral.py --era $ERA --hlt MeasFakeEl23 --wp tight --syst RequireHeavyTag >> output/$ERA/MeasFakeEl23/fitresult.tight.RequireHeavyTag.csv
