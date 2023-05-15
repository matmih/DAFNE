computeFeaturesAndDatasetsRJS<-function(putanja,putanjaTest,ime,outputPutanja){
  require("Boruta")
  require("foreign")
  
  data<-read.arff(putanja)
  dataTest<-read.arff(putanjaTest)
  
  indRules<-grep("r\\d+",colnames(data))
  indSupRules<-grep("rs\\d+",colnames(data))
  indSubg<-grep("S\\d+",colnames(data))
  indReds<-grep("R\\d+",colnames(data))
  indAll<-c(indRules,indSupRules,indSubg,indReds)
  indSSR<-c(indSubg,indReds, indSupRules)
  indSR<-c(indRules,indSubg,indReds);
  indSbg<-c(indRules,indSupRules,indReds);
  indRds<-c(indRules,indSupRules,indSubg);
  
  percentages<-c(length(indRules),length(indSupRules),length(indSubg),length(indReds),length(indRules),length(indSupRules),length(indSubg),length(indReds));
  
  dataOrig<-data[,-indAll]
  dataOrigRules<-data[,-indSSR]
  dataOrigReds<-data[,-indRds]
  dataOrigSRules<-data[,-indSR]
  dataOrigSubg<-data[,-indSbg]
  
  write.arff(dataOrig,paste(paste(paste(outputPutanja,"Original",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigSRules,paste(paste(paste(outputPutanja,"OrigSRules",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigRules,paste(paste(paste(outputPutanja,"OrigRules",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigSubg,paste(paste(paste(outputPutanja,"OrigSubg",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigReds,paste(paste(paste(outputPutanja,"OrigReds",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(data,paste(paste(paste(outputPutanja,"OrigAll",sep = ""),ime,sep = ""),".arff", sep = ""))
  
  dataOrig<-dataTest[,-indAll]
  dataOrigRules<-dataTest[,-indSSR]
  dataOrigReds<-dataTest[,-indRds]
  dataOrigSRules<-dataTest[,-indSR]
  dataOrigSubg<-dataTest[,-indSbg]
  
  
  write.arff(dataOrig,paste(paste(paste(outputPutanja,"OriginalTest",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigSRules,paste(paste(paste(outputPutanja,"OrigSRulesTest",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigRules,paste(paste(paste(outputPutanja,"OrigRulesTest",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigSubg,paste(paste(paste(outputPutanja,"OrigSubgTest",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigReds,paste(paste(paste(outputPutanja,"OrigRedsTest",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataTest,paste(paste(paste(outputPutanja,"OrigAllTest",sep = ""),ime,sep = ""),".arff", sep = ""))

  data1<-data[,-1];
print(paste(names(data[dim(data)[2]]), "~."));  
 # Boruta.Test<-Boruta(Rings ~.,data = data1,doTrace = 2,ntree = 500)#staviti 2000
  Boruta.Test<-Boruta(as.formula(paste(names(data[dim(data)[2]]), "~.")),data = data1,doTrace = 2,ntree = 2000)
 
  atstat<-attStats(Boruta.Test)
  df<-data.frame(atstat);
  write.table(df,paste(paste(paste(outputPutanja,"FeatureImportanceBoruta"),ime,sep = ""),".txt"),sep="\t")
  indUsef<-which(atstat[,6] == "Confirmed")
  indUsef<-indUsef+1
  indUsef<-c(1,indUsef)
  indUsef<-c(indUsef,length(colnames(data)))
  dataAll<-data[,indUsef]
  dataAllTest<-dataTest[,indUsef]
  
  indRules<-grep("r\\d+",colnames(dataAll))
  indSupRules<-grep("rs\\d+",colnames(dataAll))
  indSubg<-grep("S\\d+",colnames(dataAll))
  indReds<-grep("R\\d+",colnames(dataAll))
  indAll<-c(indRules,indSupRules,indSubg,indReds)
  indSSR<-c(indSubg,indReds, indSupRules)
  indSR<-c(indRules,indSubg,indReds);
  indSbg<-c(indRules,indSupRules,indReds);
  indRds<-c(indRules,indSupRules,indSubg);
  
  percentages[1]<-length(indRules)/percentages[1];
  percentages[2]<-length(indSupRules)/percentages[2];
  percentages[3]<-length(indSubg)/percentages[3];
  percentages[4]<-length(indReds)/percentages[4];
  
  if(length(indAll)>0)
  dataOrig<-dataAll[,-indAll]
  else dataOrig<-dataAll
  if(length(indSSR)>0)
  dataOrigRules<-dataAll[,-indSSR]
  else dataOrigRules<-dataOrig
  if(length(indRds)>0)
  dataOrigReds<-dataAll[,-indRds]
  else dataOrigReds<-dataOrig
  if(length(indSR)>0)
  dataOrigSRules<-dataAll[,-indSR]
  else dataOrigSRules<-dataOrig
  if(length(indSbg)>0)
  dataOrigSubg<-dataAll[,-indSbg]
  else dataOrigSubg<-dataOrig
  
  write.arff(dataOrig,paste(paste(paste(outputPutanja,"OriginalBoruta",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigSRules,paste(paste(paste(outputPutanja,"OrigSRulesBoruta",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigRules,paste(paste(paste(outputPutanja,"OrigRulesBoruta",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigSubg,paste(paste(paste(outputPutanja,"OrigSubgBoruta",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigReds,paste(paste(paste(outputPutanja,"OrigRedsBoruta",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataAll,paste(paste(paste(outputPutanja,"OrigAllBoruta",sep = ""),ime,sep = ""),".arff", sep = ""))
  
  if(length(indAll)>0)
  dataOrig<-dataAllTest[,-indAll]
  else dataOrig<-dataAllTest
  if(length(indSSR)>0)
  dataOrigRules<-dataAllTest[,-indSSR]
  else dataOrigRules<-dataOrig
  if(length(indRds)>0)
  dataOrigReds<-dataAllTest[,-indRds]
  else dataOrigReds<-dataOrig
  if(length(indSR)>0)
  dataOrigSRules<-dataAllTest[,-indSR]
  else dataOrigSRules<-dataOrig
  if(length(indSbg)>0)
  dataOrigSubg<-dataAllTest[,-indSbg]
  else dataOrigSubg<-dataOrig
  
  write.arff(dataOrig,paste(paste(paste(outputPutanja,"OriginalBorutaTest",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigSRules,paste(paste(paste(outputPutanja,"OrigSRulesBorutaTest",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigRules,paste(paste(paste(outputPutanja,"OrigRulesBorutaTest",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigSubg,paste(paste(paste(outputPutanja,"OrigSubgBorutaTest",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigReds,paste(paste(paste(outputPutanja,"OrigRedsBorutaTest",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataAllTest,paste(paste(paste(outputPutanja,"OrigAllBorutaTest",sep = ""),ime,sep = ""),".arff", sep = ""))

  #Boruta on test set as validation
  data1Test<-dataTest[,-1];
  print(paste(names(data[dim(data)[2]]), "~."));  
  # Boruta.Test<-Boruta(Rings ~.,data = data1,doTrace = 2,ntree = 500)#staviti 2000
  Boruta.Test<-Boruta(as.formula(paste(names(data[dim(data)[2]]), "~.")),data = data1Test,doTrace = 2,ntree = 2000)
  
  atstatTest<-attStats(Boruta.Test)
  dfTest<-data.frame(atstatTest);
  write.table(dfTest,paste(paste(paste(outputPutanja,"FeatureImportanceBorutaTest"),ime,sep = ""),".txt"),sep="\t")
  indUsefT<-which(atstatTest[,6] == "Confirmed")
  indUsefT<-indUsefT+1
  indUsefT<-c(1,indUsefT)
  indUsefT<-c(indUsefT,length(colnames(data)))
  dataAllT<-data[,indUsefT]
  indRulesT<-grep("r\\d+",colnames(dataAllT))
  indSupRulesT<-grep("rs\\d+",colnames(dataAllT))
  indSubgT<-grep("S\\d+",colnames(dataAllT))
  indRedsT<-grep("R\\d+",colnames(dataAllT))
  
  percentages[5]<-length(indRulesT)/percentages[5];
  percentages[6]<-length(indSupRulesT)/percentages[6];
  percentages[7]<-length(indSubgT)/percentages[7];
  percentages[8]<-length(indRedsT)/percentages[8];
  write(percentages,file = paste(paste(paste(outputPutanja,"Percentages",sep = ""),ime, sep = ""),".txt",sep = ""), sep = " ",ncolumns = 4);
  #koristiti varSelRF umjesto FSelectora (istoimena funkcija - slicno kao u DS èlanku)
  require("varSelRF")
  dataAll1<-dataAll[,-1]
  C<-factor(dataAll1[,length(colnames(dataAll1))])
  dataAll1<-dataAll1[,-length(colnames(dataAll1))]
  rf.vs1 <- varSelRF(dataAll1, C, vars.drop.frac = 0.2, c.sd = 1)
  df<-data.frame(rf.vs1$selected.vars)
  #View(dataAll1)
  #subsetA <- cfs(Rings~., dataAll1)
  #df<-data.frame(subsetA);
  write.table(df,paste(paste(paste(outputPutanja,"FeatureImportanceNonRedundant"),ime,sep = ""),".txt"),sep="\t")
  #indNonRed<-match(subsetA,colnames(dataAll1))
  indNonRed<-match(rf.vs1$selected.vars,colnames(dataAll1))
  indNonRed<-indNonRed+1
  indNonRed<-c(1,indNonRed)
  indNonRed<-c(indNonRed,length(colnames(dataAll)))
  
  dataAllNR<-dataAll[,indNonRed]
  dataAllNRTest<-dataAllTest[,indNonRed]
  write.arff(dataAllNR,paste(paste(paste(outputPutanja,"OrigAllBNR",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataAllNRTest,paste(paste(paste(outputPutanja,"OrigAllBNRTest",sep = ""),ime,sep = ""),".arff", sep = ""))
  
  indRules<-grep("r\\d+",colnames(dataAllNR))
  indSupRules<-grep("rs\\d+",colnames(dataAllNR))
  indSubg<-grep("S\\d+",colnames(dataAllNR))
  indReds<-grep("R\\d+",colnames(dataAllNR))
  indAll<-c(indRules,indSupRules,indSubg,indReds)
  indSSR<-c(indSubg,indReds, indSupRules)
  indSR<-c(indRules,indSubg,indReds);
  indSbg<-c(indRules,indSupRules,indReds);
  indRds<-c(indRules,indSupRules,indSubg);

  if(length(indAll)>0)
  dataOrig<-dataAllNR[,-indAll]
  else dataOrig<-dataAllNR
  if(length(indRules)>0 && length(indSSR)>0)
  dataOrigRules<-dataAllNR[,-indSSR]
  else if(length(indRules)>0)
  dataOrigRules<-dataAllNR
  else dataOrigRules<-dataOrig
  if(length(indReds)>0 && length(indRds)>0)
    dataOrigReds<-dataAllNR[,-indRds]
  else if(length(indReds)>0)
    dataOrigReds<-dataAllNR
  else dataOrigReds<-dataOrig
  if(length(indSupRules)>0 && length(indSR)>0)
     dataOrigSRules<-dataAllNR[,-indSR]
  else if(length(indSupRules)>0) dataOrigSRules<-dataAllNR
  else  dataOrigSRules<-dataOrig
  if(length(indSubg)>0 && length(indSbg)>0)
      dataOrigSubg<-dataAllNR[,-indSbg]
  else if(length(indSubg)>0) dataOrigSubg<-dataAllNR
  else dataOrigSubg<-dataOrig
  
  write.arff(dataOrig,paste(paste(paste(outputPutanja,"OriginalBNR",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigSRules,paste(paste(paste(outputPutanja,"OrigSRulesBNR",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigRules,paste(paste(paste(outputPutanja,"OrigRulesBNR",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigSubg,paste(paste(paste(outputPutanja,"OrigSubgBNR",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigReds,paste(paste(paste(outputPutanja,"OrigRedsBNR",sep = ""),ime,sep = ""),".arff", sep = ""))
 
  
  if(length(indAll)>0)
    dataOrig<-dataAllNRTest[,-indAll]
  else dataOrig<-dataAllNRTest
  if(length(indRules)>0 && length(indSSR)>0)
    dataOrigRules<-dataAllNRTest[,-indSSR]
  else if(length(indRules)>0)
    dataOrigRules<-dataAllNRTest
  else dataOrigRules<-dataOrig
  if(length(indReds)>0 && length(indRds)>0)
    dataOrigReds<-dataAllNRTest[,-indRds]
  else if(length(indReds)>0)
    dataOrigReds<-dataAllNRTest
  else dataOrigReds<-dataOrig
  if(length(indSupRules)>0 && length(indSR)>0)
    dataOrigSRules<-dataAllNRTest[,-indSR]
  else if(length(indSupRules)>0) dataOrigSRules<-dataAllNRTest
  else  dataOrigSRules<-dataOrig
  if(length(indSubg)>0 && length(indSbg)>0)
    dataOrigSubg<-dataAllNRTest[,-indSbg]
  else if(length(indSubg)>0) dataOrigSubg<-dataAllNRTest
  else dataOrigSubg<-dataOrig
  
 # if(length(indAll)>0)
#  dataOrig<-dataAllNRTest[,-indAll]
#  else dataOrig<-dataAllNRTest
#  if(length(indRules)>0)
#    dataOrigRules<-dataAllNRTest[,-indSSR]
#  else dataOrigRules<-dataOrig
#  if(length(indReds)>0)
#    dataOrigReds<-dataAllNRTest[,-indRds]
#  else dataOrigReds<-dataOrig
#  if(length(indSupRules)>0)
#    dataOrigSRules<-dataAllNRTest[,-indSR]
#  else  dataOrigSRules<-dataOrig
#  if(length(indSubg)>0)
#    dataOrigSubg<-dataAllNRTest[,-indSbg]
#  else dataOrigSubg<-dataOrig
  
  write.arff(dataOrig,paste(paste(paste(outputPutanja,"OriginalBNRTest",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigSRules,paste(paste(paste(outputPutanja,"OrigSRulesBNRTest",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigRules,paste(paste(paste(outputPutanja,"OrigRulesBNRTest",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigSubg,paste(paste(paste(outputPutanja,"OrigSubgBNRTest",sep = ""),ime,sep = ""),".arff", sep = ""))
  write.arff(dataOrigReds,paste(paste(paste(outputPutanja,"OrigRedsBNRTest",sep = ""),ime,sep = ""),".arff", sep = ""))
  
}