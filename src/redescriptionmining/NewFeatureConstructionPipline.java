/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package redescriptionmining;

import gnu.trove.iterator.TIntIterator;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Random;
import org.apache.commons.compress.utils.Charsets;
import org.rosuda.JRI.Rengine;
import parsers.RuleSet;
import sgd.CN2SD;
import sgd.DataSet;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.KStar;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.LMT;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Matej
 */
public class NewFeatureConstructionPipline {
    
    static Instances loadMappingAndTargetIndex(ApplicationSettings appset,  Mappings fid,  Mappings fidFull,  Mappings fidTest , WekaDatasetLoader wdl, HashMap<String,Integer> targets,HashMap<Integer,Integer> targetsToIndex){
        Instances data = null;
        
        if(appset.system.equals("windows")){
                fid.createIndex(appset.outFolderPath+"\\Jinput.arff"); 
                data = wdl.loadDataset(appset.outFolderPath+"\\Jinput.arff");
                data.setClassIndex(data.numAttributes()-1);
           }
           else{
                fid.createIndex(appset.outFolderPath+"/Jinput.arff");
                data = wdl.loadDataset(appset.outFolderPath+"/Jinput.arff");
                data.setClassIndex(data.numAttributes()-1);//class indeks must be last atribute
           }
           
           int tcount=0;
           
           //load target info
           for(int i=0;i<data.numInstances();i++){
               targets.put(data.get(i).stringValue(0),(int)data.get(i).value(data.get(i).classIndex()));
               if(!targetsToIndex.containsKey((int)data.get(i).value(data.get(i).classIndex()))){
                   targetsToIndex.put((int)data.get(i).value(data.get(i).classIndex()), tcount++);
               }
           }
           
           if(appset.useSplitTesting==true){
               if(appset.system.equals("windows")){
                        fidFull.createIndex(appset.outFolderPath+"\\Jinput.arff");
                        fidTest.createIndex(appset.outFolderPath+"\\Jinput.arff");
               }
               else{
                   fidFull.createIndex(appset.outFolderPath+"/Jinput.arff");
                   fidTest.createIndex(appset.outFolderPath+"/Jinput.arff");
               }
           }
           return data;
    }
    
    static HashMap<ArrayList<Instances>,ArrayList<DataSetCreator>> createTrainTest(ApplicationSettings appset, Mappings fid, Instances data, InstancesFilter insfilt, DataSetCreator datJ, DataSetCreator datJFull, DataSetCreator datJTest, DataSetCreator datJValid, DataSetCreator datJTrainValid){
        
        Instances dataTrain = null, dataValidation = null, dataTrainValidation = null , dataTest = null, dataFullComputation = null;
        ArrayList<Instances> ret = new ArrayList<>();
        ArrayList<DataSetCreator> ret1 = new ArrayList<>();
        HashMap<ArrayList<Instances>,ArrayList<DataSetCreator>> retMap = new HashMap<>();
        DataSetCreator datJFullC = null;
        
        insfilt = new InstancesFilter(data);
               try{
                   Instances tmp;
                    insfilt.removeStratifiedFoldsFilter(1, 5, true);
                    dataTest = insfilt.getFilteredInstances(); //test set

                    System.out.println("NAbefore: "+dataTest.numAttributes());
                    Remove rm = new Remove();
                    int indC1 = dataTest.numAttributes()/2+1;
                    
                    rm.setAttributeIndices((indC1+1)+"-"+(dataTest.numAttributes()));
                    rm.setInputFormat(dataTest);
                    tmp = Filter.useFilter(dataTest, rm);
                    
                     String outputFilenamePred = "";
                    if(appset.system.equals("windows"))
                        outputFilenamePred= appset.outFolderPath+"\\Test.arff";
                    else outputFilenamePred= appset.outFolderPath+"/Test.arff";
                     ConverterUtils.DataSink.write(outputFilenamePred, tmp); //write the predictive test set
                    
                    rm.setAttributeIndices(indC1+","+(dataTest.numAttributes()));
                    rm.setInputFormat(dataTest);
                    dataTest = Filter.useFilter(dataTest, rm);

                    System.out.println("NAafter: "+dataTest.numAttributes());
                    insfilt = new InstancesFilter(data);
                    insfilt.removeStratifiedFoldsFilter(1,5,false);
                    dataTrainValidation = insfilt.getFilteredInstances();
                    
                    rm.setAttributeIndices((indC1+1)+"-"+(dataTrainValidation.numAttributes()));
                    rm.setInputFormat(dataTrainValidation);
                    tmp = Filter.useFilter(dataTrainValidation, rm);
                    
                     if(appset.system.equals("windows"))
                        outputFilenamePred= appset.outFolderPath+"\\TrainValidation.arff";
                    else outputFilenamePred= appset.outFolderPath+"/TrainValidation.arff";
                     ConverterUtils.DataSink.write(outputFilenamePred, tmp); //write the predictive test set
                    
                      rm.setAttributeIndices(indC1+","+(dataTrainValidation.numAttributes()));
                     insfilt = new InstancesFilter(dataTrainValidation);
                     insfilt.removeStratifiedFoldsFilter(1,4,true);
                     dataValidation = insfilt.getFilteredInstances();
                     
                     insfilt = new InstancesFilter(dataTrainValidation);
                     insfilt.removeStratifiedFoldsFilter(1,4,false);
                     dataTrain = insfilt.getFilteredInstances();
                    
                      rm.setInputFormat(dataTrainValidation);
                     dataTrainValidation = Filter.useFilter(dataTrainValidation, rm);
                     
                      rm.setInputFormat(dataValidation);
                     dataValidation = Filter.useFilter(dataValidation, rm);
                     
                      rm.setInputFormat(dataTrain);
                     dataTrain = Filter.useFilter(dataTrain, rm);
                     
                     rm.setInputFormat(data);
                     dataFullComputation = Filter.useFilter(data, rm);
                     
                     String outputFilename = "";
                    if(appset.system.equals("windows"))
                        outputFilename= appset.outFolderPath+"\\JinputTest.arff";
                    else outputFilename= appset.outFolderPath+"/JinputTest.arff";
                     ConverterUtils.DataSink.write(outputFilename, dataTest);
                     if(appset.system.equals("windows"))
                        outputFilename = appset.outFolderPath+"\\JinputTrainValidation.arff";
                     else  outputFilename = appset.outFolderPath+"/JinputTrainValidation.arff";
                     ConverterUtils.DataSink.write(outputFilename, dataTrainValidation);
                     
                     if(appset.system.equals("windows"))
                                outputFilename = appset.outFolderPath+"\\JinputValidation.arff";
                     else  outputFilename = appset.outFolderPath+"/JinputValidation.arff";
                     ConverterUtils.DataSink.write(outputFilename, dataValidation);
                     
                      if(appset.system.equals("windows"))
                             outputFilename = appset.outFolderPath+"\\JinputTrain.arff";
                      else  outputFilename = appset.outFolderPath+"/JinputTrain.arff";
                     ConverterUtils.DataSink.write(outputFilename, dataTrain);
                     
                     if(appset.system.equals("windows"))
                             outputFilename = appset.outFolderPath+"\\JinputAll.arff";
                      else  outputFilename = appset.outFolderPath+"/JinputAll.arff";
                    // BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilename),Charsets.);
                    // writer.write(dataFullComputation.toString());
                     ConverterUtils.DataSink.write(outputFilename, dataFullComputation);
                     
               }
               catch(Exception e){
                   e.printStackTrace();
               }

               datJFull=datJ;
               
                   if(appset.system.equals("windows")){
                      // datJFull = new DataSetCreator(appset.outFolderPath+"\\Jinput.arff");
                      datJFullC= new DataSetCreator(appset.outFolderPath+"\\JinputAll.arff");
                        datJ = new DataSetCreator(appset.outFolderPath+"\\JinputTrain.arff");
                        datJTest = new DataSetCreator(appset.outFolderPath+"\\JinputTest.arff");
                        datJValid = new DataSetCreator(appset.outFolderPath+"\\JinputValidation.arff");
                        datJTrainValid = new DataSetCreator(appset.outFolderPath+"\\JinputTrainValidation.arff");
                   }
                   else{
                        datJFullC= new DataSetCreator(appset.outFolderPath+"/JinputAll.arff");
                         datJ = new DataSetCreator(appset.outFolderPath+"/JinputTrain.arff");
                        datJTest = new DataSetCreator(appset.outFolderPath+"/JinputTest.arff");
                        datJValid = new DataSetCreator(appset.outFolderPath+"/JinputValidation.arff");
                        datJTrainValid = new DataSetCreator(appset.outFolderPath+"/JinputTrainValidation.arff");
                   }
                   
                    try{
                             datJFullC.readDataset();
                             datJ.readDataset();
                             datJTest.readDataset();
                             datJValid.readDataset();
                             datJTrainValid.readDataset();
                         }
                     catch(IOException e){
                              e.printStackTrace();
                          }
                          /*System.out.println("datJFull index size: "+datJFull.W2indexs.size());
                          for(int zz=0;zz<datJFull.W2indexs.size();zz++)
                              System.out.print(datJFull.W2indexs.get(zz)+" ");
                          System.out.println();*/
                         datJFullC.W2indexs.addAll(datJFull.W2indexs);
                         datJ.W2indexs.addAll(datJFull.W2indexs);//ovo nije OK
                         datJTest.W2indexs.addAll(datJFull.W2indexs);
                         datJValid.W2indexs.addAll(datJFull.W2indexs);   
                         datJTrainValid.W2indexs.addAll(datJFull.W2indexs);
         
         ret1.add(datJFull); ret1.add(datJ); ret1.add(datJTest); ret1.add(datJValid); ret1.add(datJTrainValid);
         ret1.add(datJFullC);
         fid.clearMaps();
         /*if(appset.system.equals("windows"))
             fid.createIndex(appset.outFolderPath+"\\JinputTrain.arff");
         else fid.createIndex(appset.outFolderPath+"/JinputTrain.arff");*/
         
          if(appset.system.equals("windows"))
             fid.createIndex(appset.outFolderPath+"\\JinputAll.arff");
         else fid.createIndex(appset.outFolderPath+"/JinputAll.arff");
         
               ret.add(dataTrain); ret.add(dataValidation); ret.add(dataTrainValidation); ret.add(dataTest);
               retMap.put(ret, ret1);
               return retMap;
    }
    
    
    static void createDescriptiveRulesAndRedescriptions(ApplicationSettings appset, Mappings fidFull, Mappings fidTest, DataSetCreator datJTest, DataSetCreator datJFull , HashMap<String,Integer> targets,HashMap<Integer,Integer> targetsToIndex, RedescriptionSet resultSetOut, RuleReader rrResultOut){
     System.out.println("num attrs full: "+datJFull.schema.getNbAttributes());
     System.out.println("num attrs test: "+datJTest.schema.getNbAttributes());
        long startTime = System.currentTimeMillis(); 
        Random r=new Random();
        RedescriptionSet rs=new RedescriptionSet();
        RuleReader rr=new RuleReader();
        RuleReader rr1=new RuleReader();
         boolean oom[]= new boolean[1];
         
        int elemFreq[]=null;
        int attrFreq[]=null;
        ArrayList<Double> redScores=null;
        ArrayList<Double> redScoresAtt=null;
        ArrayList<Double> targetAtScore=null;
        ArrayList<Double> redDistCoverage=null;
        ArrayList<Double> redDistCoverageAt=null;
        ArrayList<Double> redDistNetwork=null;
         double Statistics[]={0.0,0.0,0.0};//previousMedian - 0, numberIterationsStable - 1, minDifference - 2
         ArrayList<Double> maxDiffScoreDistribution = null;
      
       if(appset.optimizationType == 0){
        if(appset.redesSetSizeType==1 && appset.numRetRed!=Integer.MAX_VALUE)
            appset.numInitial=appset.numRetRed;
        else{
            if(appset.numRetRed!=Integer.MAX_VALUE && appset.numRetRed!=-1)
                appset.numInitial=appset.numRetRed;
            else
                appset.numInitial=20;
        }
       }
        
        
        if(appset.optimizationType==0){
            
                   
         elemFreq=new int[datJFull.numExamples];
         attrFreq=new int[datJFull.schema.getNbAttributes()];  
            
        redScores=new ArrayList<>(appset.numInitial);
        redScoresAtt=new ArrayList<>(appset.numInitial);
        redDistCoverage=new ArrayList<>(appset.numInitial);
        redDistCoverageAt=new ArrayList<>(appset.numInitial);
        if(appset.useNetworkAsBackground==true)
              redDistNetwork=new ArrayList<>(appset.numInitial);
         targetAtScore=null;
        maxDiffScoreDistribution=new ArrayList<>(appset.numInitial);
        
        if(appset.attributeImportance!=0)
            targetAtScore = new ArrayList<>(appset.numInitial);
        
        for(int z=0;z<appset.numInitial;z++){
            redScores.add(Double.NaN);
            redScoresAtt.add(Double.NaN);
            redDistCoverage.add(Double.NaN);
            redDistCoverageAt.add(Double.NaN);
            maxDiffScoreDistribution.add(Double.NaN);
            if(appset.useNetworkAsBackground==true)
                 redDistNetwork.add(Double.NaN);
            if(appset.attributeImportance!=0)
                targetAtScore.add(Double.NaN);
        }   
      }
                    
        NHMCDistanceMatrix nclMatInit=null;
        if(appset.distanceFilePaths.size()>0){
            nclMatInit=new NHMCDistanceMatrix(datJFull.numExamples,appset);
            nclMatInit.loadDistance(new File(appset.distanceFilePaths.get(0)), fidFull);
            if(appset.distanceFilePaths.size()>0){
             nclMatInit.resetFile(new File(appset.outFolderPath+"\\distances.csv"));
             nclMatInit.writeToFile(new File(appset.outFolderPath+"\\distances.csv"), fidFull,appset);
            }
            else{
                nclMatInit.resetFile(new File(appset.outFolderPath+"/distances.csv"));
                nclMatInit.writeToFile(new File(appset.outFolderPath+"/distances.csv"), fidFull,appset);
            }
             nclMatInit=null;
        }
        
        for(int runTest=0;runTest<appset.numRandomRestarts;runTest++){  

          DataSetCreator datJInit=null;
          
       if(!appset.useSplitTesting){ 
        if(appset.initClusteringFileName.equals("")){
            if(appset.system.equals("windows")){
                datJInit = new DataSetCreator(appset.outFolderPath+"\\JinputAll.arff");
            }
            else{
                datJInit = new DataSetCreator(appset.outFolderPath+"/JinputAll.arff");
            }
        }
        else{
                     if(appset.system.equals("windows"))
                            datJInit = new DataSetCreator(appset.outFolderPath+"\\"+appset.initClusteringFileName);
                     else
                            datJInit = new DataSetCreator(/*appset.outFolderPath+"/"+*/appset.initClusteringFileName);               
            }
       }
       else{
           if(appset.trainFileName.equals("") || appset.testFileName.equals("")){
            if(appset.system.equals("windows"))
                datJInit = new DataSetCreator(appset.outFolderPath+"\\JinputAll.arff");
            else
                datJInit = new DataSetCreator(appset.outFolderPath+"/JinputAll.arff");
           }
           else{
               if(appset.system.equals("windows"))
                datJInit = new DataSetCreator(appset.outFolderPath+"\\"+appset.trainFileName);
            else
                datJInit = new DataSetCreator(appset.outFolderPath+"/"+appset.trainFileName);
           }
       }
        
                try{
        datJInit.readDataset();
        }
        catch(IOException e){
            e.printStackTrace();
        }
        
        datJInit.W2indexs.addAll(datJFull.W2indexs);
        
        if(appset.initClusteringFileName.equals(""))
            datJInit.initialClusteringGen2(appset.outFolderPath,appset,datJFull.schema.getNbDescriptiveAttributes(),r);
        
        SettingsReader initSettings=new SettingsReader();
        
        if(appset.initClusteringFileName.equals(""))
             if(appset.system.equals("windows"))
                 initSettings.setDataFilePath(appset.outFolderPath+"\\JinputAll1.arff");
             else
                  initSettings.setDataFilePath(appset.outFolderPath+"/JinputAll1.arff");
        else{
            if(appset.system.equals("windows"))
                 initSettings.setDataFilePath(appset.outFolderPath+"\\"+appset.initClusteringFileName);
            else
                initSettings.setDataFilePath(/*appset.outFolderPath+"/"+*/appset.initClusteringFileName);
        }

        if(appset.system.equals("windows"))
             initSettings.setPath(appset.outFolderPath+"\\view1.s");
        else
             initSettings.setPath(appset.outFolderPath+"/view1.s");

        if(appset.useNC.get(0) == false)
             initSettings.createInitialSettingsGen(0, 3, datJFull.W2indexs.get(0), datJFull.schema.getNbAttributes(), appset,1);
        else
             initSettings.createInitialSettingsGen(0, 4, datJFull.W2indexs.get(0), datJFull.schema.getNbAttributes(), appset,1);
            
        ClusProcessExecutor exec=new ClusProcessExecutor();

        //RunInitW1S1
        exec.run(appset.javaPath,appset.clusPath ,appset.outFolderPath,"view1.s",0, appset.clusteringMemory);//was 1 before for rules

          String input1="";
          if(appset.system.equals("windows"))
             input1=appset.outFolderPath+"\\view1.out";
          else
              input1=appset.outFolderPath+"/view1.out"; 
           
           rr1.extractRules(input1,fidFull,datJInit,appset);
           
         //reading arff file
         
           if(appset.distanceFilePaths.size()>1){
            nclMatInit=new NHMCDistanceMatrix(datJFull.numExamples,appset);
            nclMatInit.loadDistance(new File(appset.distanceFilePaths.get(1)), fidFull);
            if(appset.system.equals("windows")){
             nclMatInit.resetFile(new File(appset.outFolderPath+"\\distances.csv"));
             nclMatInit.writeToFile(new File(appset.outFolderPath+"\\distances.csv"), fidFull,appset);
            }
            else{
                nclMatInit.resetFile(new File(appset.outFolderPath+"/distances.csv"));
                nclMatInit.writeToFile(new File(appset.outFolderPath+"/distances.csv"), fidFull,appset);
            }
             nclMatInit=null;
        }
       
        SettingsReader set=null;
        SettingsReader set1=null;
        SettingsReader setF=null;
        SettingsReader setF1=null;
       
        //RunInitW1S2
        if(appset.system.equals("windows")){
            initSettings.setPath(appset.outFolderPath+"\\view2.s");
        }
        else{
           initSettings.setPath(appset.outFolderPath+"/view2.s"); 
        }
        //initSettings.createInitialSettings1(2, datJ.W2indexs.get(0), datJInit.schema.getNbAttributes(), appset);
        if(datJFull.W2indexs.size()>1)
            initSettings.createInitialSettingsGen(1, datJFull.W2indexs.get(0)+1, datJFull.W2indexs.get(1), datJFull.schema.getNbAttributes(), appset,1);
        else
            initSettings.createInitialSettingsGen(1, datJFull.W2indexs.get(0)+1, datJInit.schema.getNbAttributes(), datJFull.schema.getNbAttributes(), appset,1);
        
        exec.run(appset.javaPath,appset.clusPath, appset.outFolderPath, "view2.s", 0,appset.clusteringMemory);//was 1 before
        System.out.println("Process 1 side 2 finished!");

        //read the rules obtained from first attribute set
       if(appset.system.equals("windows"))
        input1=appset.outFolderPath+"\\view2.out";
       else
           input1=appset.outFolderPath+"/view2.out";
        rr.extractRules(input1,fidFull,datJInit,appset);
           
           FileDeleter delTmp=new FileDeleter();
           if(appset.system.equals("windows"))
                delTmp.setPath(appset.outFolderPath+"\\JinputAll1.arff");
           else
               delTmp.setPath(appset.outFolderPath+"/JinputAll1.arff");
           delTmp.delete();
        
          if(appset.system.equals("windows"))
           initSettings.setPath(appset.outFolderPath+"\\view1.s");
          else
              initSettings.setPath(appset.outFolderPath+"/view1.s");

          if(appset.system.equals("windows")) 
             initSettings.setPath(appset.outFolderPath+"\\view2.s");
          else
             initSettings.setPath(appset.outFolderPath+"/view2.s");
       
           datJInit=null;        
           
        int leftSide=1, rightSide=0;//set left to 1 when computing lf, otherwise right
        int leftSide1=0, rightSide1=1; //left, right side for Side 2
        int it=0;
        Jacard js=new Jacard();
        Jacard jsN[]=new Jacard[3];
        
        for(int i=0;i<jsN.length;i++)
            jsN[i]=new Jacard();
       
        int newRedescriptions=1;
        int numIter=0;
        int RunInd=0;
       
        int naex=datJFull.numExamples;
        
        //add arrayList of view rules
        ArrayList<RuleReader> readers=new ArrayList<>();
        int oldRIndex[]={0};
        
        NHMCDistanceMatrix nclMat=null;
        if((appset.distanceFilePaths.size()>0 || appset.useNC.get(0)==true) && appset.networkInit==false)
            nclMat=new NHMCDistanceMatrix(datJFull.numExamples,appset);
        NHMCDistanceMatrix nclMat1=null;
        if((appset.distanceFilePaths.size()>1 || appset.useNC.get(1)==true) && appset.networkInit==false)
            nclMat1=new NHMCDistanceMatrix(datJFull.numExamples,appset);
       
        if(appset.useNetworkAsBackground==true)
            appset.networkInit=false;
        
        if(appset.useNC.size()>=2 && appset.useNC.get(1) == true){
            if(appset.system.equals("windows")) 
                initSettings.setPath(appset.outFolderPath+"\\view2.s");
            else
                initSettings.setPath(appset.outFolderPath+"/view2.s");
         if(appset.useNC.size()>2)
            initSettings.createInitialSettingsGenN(1, datJFull.W2indexs.get(0)+1, datJFull.W2indexs.get(1), datJFull.schema.getNbAttributes(), appset);
         else
            initSettings.createInitialSettingsGenN(1, datJFull.W2indexs.get(0)+1, datJFull.schema.getNbAttributes()+1, datJFull.schema.getNbAttributes(), appset); 
        }
        if(appset.useNC.size()>1 && appset.useNC.get(0) == true){
             if(appset.system.equals("windows")) 
                initSettings.setPath(appset.outFolderPath+"\\view1.s");
             else
                initSettings.setPath(appset.outFolderPath+"/view1.s");
         initSettings.createInitialSettingsGenN(1, 4, datJFull.W2indexs.get(0), datJFull.schema.getNbAttributes(), appset);
        }

        while(newRedescriptions!=0 && RunInd<appset.numIterations){
            
       DataSetCreator dsc=null;//new DataSetCreator(appset.outFolderPath+"\\Jinput.arff");
       DataSetCreator dsc1=null;//new DataSetCreator(appset.outFolderPath+"\\Jinput.arff");
       
       rr.setSize();
       rr1.setSize();
       
       int nARules=0, nARules1=0;
       int oldIndexRR=rr.newRuleIndex;
       int oldIndexRR1=rr1.newRuleIndex;
       System.out.println("OOIndRR: "+oldIndexRR);
       System.out.println("OOIndRR1: "+oldIndexRR1);
       int endIndexRR=0, endIndexRR1=0;
             newRedescriptions=0;
            System.out.println("Iteration: "+(++numIter));

             //do rule creation with various generality levels
        double percentage[]=new double[]{0,0.2,0.4,0.6,0.8,1.0};

         int numBins=0;
        int Size=Math.max(rr.rules.size()-oldIndexRR, rr1.rules.size()-oldIndexRR1);
        if(Size%appset.numTargets==0)
            numBins=Size/appset.numTargets;
        else numBins=Size/appset.numTargets+1;
        
        for(int z=0;z<numBins;z++){//percentage.length-1;z++){

            nARules=0; nARules1=0;
            double startPerc=0;//percentage[z];
            double endPerc=1;//percentage[z+1];
            int minCovElements[]=new int[]{0}, minCovElements1[]=new int[]{0};
            int maxCovElements[]=new int[]{0}, maxCovElements1[]=new int[]{0};

            System.out.println("startPerc: "+startPerc);
            System.out.println("endPerc: "+endPerc);

            int cuttof=0,cuttof1=0;

            if(z==0){
                endIndexRR=rr.rules.size();
                endIndexRR1=rr1.rules.size();
               
                //compute network things only here
                if(appset.useNC.get(1)==true && appset.networkInit==false && appset.useNetworkAsBackground==false){
                    nclMat.reset(appset);
                    nclMat1.reset(appset);
                if(leftSide==1){
                    if(appset.distanceFilePaths.size()>=1){
                             nclMat.loadDistance(new File(appset.distanceFilePaths.get(1)), fidFull);
                    }
                     else if(appset.computeDMfromRules==true){
                             nclMat.computeDistanceMatrix(rr1, fidFull, appset.maxDistance, datJFull.numExamples);
                     }
                   }
                }
                 if(appset.useNC.get(0)==true && appset.networkInit==false && appset.useNetworkAsBackground==false){
                 if(rightSide==1){
                    if(appset.distanceFilePaths.size()>=2){
                             nclMat.loadDistance(new File(appset.distanceFilePaths.get(0)), fidFull);
                    }
                     else if(appset.computeDMfromRules==true){
                             nclMat.computeDistanceMatrix(rr, fidFull, appset.maxDistance, datJFull.numExamples);
                     }
                   }
                 }
                  if(appset.useNC.get(1)==true && appset.networkInit==false && appset.useNetworkAsBackground==false){
                if(leftSide1==1){
                    if(appset.distanceFilePaths.size()>=1){
                             nclMat1.loadDistance(new File(appset.distanceFilePaths.get(1)), fidFull);
                    }
                     else if(appset.computeDMfromRules==true){
                             nclMat1.computeDistanceMatrix(rr1, fidFull, appset.maxDistance, datJFull.numExamples);
                     }
                }
                  }
                   if(appset.useNC.get(0)==true && appset.networkInit==false && appset.useNetworkAsBackground==false){
                    if(rightSide1==1){
                    if(appset.distanceFilePaths.size()>=2){
                             nclMat1.loadDistance(new File(appset.distanceFilePaths.get(0)), fidFull);
                            // nclMat1.writeToFile(new File(appset.outFolderPath+"\\distance.csv"), fid);
                    }
                     else if(appset.computeDMfromRules==true){
                             nclMat1.computeDistanceMatrix(rr, fidFull, appset.maxDistance, datJFull.numExamples);
                     }
                  }
                }
            }
            
         if(!appset.useSplitTesting==true){    
          if(appset.system.equals("windows")){         
                dsc=new DataSetCreator(appset.outFolderPath+"\\Jinput.arff");
                dsc1=new DataSetCreator(appset.outFolderPath+"\\Jinput.arff");
          }
          else{
                dsc=new DataSetCreator(appset.outFolderPath+"/Jinput.arff");
                dsc1=new DataSetCreator(appset.outFolderPath+"/Jinput.arff");
          }
         }
         else{
              if(appset.trainFileName.equals("") || appset.testFileName.equals("")){
                     if(appset.system.equals("windows")){    
                             dsc=new DataSetCreator(appset.outFolderPath+"\\JinputAll.arff");
                             dsc1=new DataSetCreator(appset.outFolderPath+"\\JinputAll.arff");
                         }
                     else{
                             dsc=new DataSetCreator(appset.outFolderPath+"/JinputAll.arff");
                             dsc1=new DataSetCreator(appset.outFolderPath+"/JinputAll.arff"); 
                }
             }
              else{
                  if(appset.system.equals("windows")){    
                             dsc=new DataSetCreator(appset.outFolderPath+"\\"+appset.trainFileName);
                             dsc1=new DataSetCreator(appset.outFolderPath+"\\"+appset.trainFileName);
                         }
                     else{
                             dsc=new DataSetCreator(appset.outFolderPath+"/"+appset.trainFileName);
                             dsc1=new DataSetCreator(appset.outFolderPath+"/"+appset.trainFileName); 
                }
              }
         }
    
            try{
        dsc.readDataset();
        }
        catch(IOException e){
            e.printStackTrace();
        }
            naex=dsc.data.getNbRows();
        //read dataset for cicle 2
        try{
        dsc1.readDataset();
        }
        catch(IOException e){
            e.printStackTrace();
        }
            
            //create and modify settings for cicle 1
          if(leftSide==1 && (endIndexRR1-oldIndexRR1)>z*appset.numTargets){
              //createSettings
            if(appset.system.equals("windows")){    
             set=new SettingsReader(appset.outFolderPath+"\\view2tmp.s",appset.outFolderPath+"\\view2.s");
             set.setDataFilePath(appset.outFolderPath+"\\Jinputnew.arff");
            }
            else{
                 set=new SettingsReader(appset.outFolderPath+"/view2tmp.s",appset.outFolderPath+"/view2.s");
                 set.setDataFilePath(appset.outFolderPath+"/Jinputnew.arff"); 
            }
             if(appset.numSupplementTrees>0){
                 if(appset.system.equals("windows")){
                     setF=new SettingsReader(appset.outFolderPath+"\\view2tmpF.s",appset.outFolderPath+"\\view2.s");
                     setF.setDataFilePath(appset.outFolderPath+"\\Jinputnew.arff");
                 }
                 else{
                     setF=new SettingsReader(appset.outFolderPath+"/view2tmpF.s",appset.outFolderPath+"/view2.s");
                     setF.setDataFilePath(appset.outFolderPath+"/Jinputnew.arff");
                 }
             }
             
             int endTmp=0;
             if((z+1)*appset.numTargets>(endIndexRR1-oldIndexRR1))
                 endTmp=endIndexRR1;
             else endTmp=(z+1)*appset.numTargets+oldIndexRR1;
             int startIndexRR1=oldIndexRR1+z*appset.numTargets;
             
             for(int i=startIndexRR1;i<endTmp;i++) //do on the fly when reading rules
                    if( rr1.rules.get(i).elements.size()>=appset.minSupport) //do parameters analysis in this step
                        nARules++;
             set.ModifySettings(nARules,dsc.schema.getNbAttributes());
             if(appset.numSupplementTrees>0)
                setF.ModifySettingsF(nARules,dsc.schema.getNbAttributes(),appset);
          }
          else if(rightSide==1 && (endIndexRR-oldIndexRR)>z*appset.numTargets){
              if(appset.system.equals("windows")){
                 set=new SettingsReader(appset.outFolderPath+"\\view1tmp.s",appset.outFolderPath+"\\view1.s");
                 set.setDataFilePath(appset.outFolderPath+"\\Jinputnew.arff");
                    }
              else{
                 set=new SettingsReader(appset.outFolderPath+"/view1tmp.s",appset.outFolderPath+"/view1.s");
                 set.setDataFilePath(appset.outFolderPath+"/Jinputnew.arff");  
              }
                 if(appset.numSupplementTrees>0){
                     if(appset.system.equals("windows")){
                        setF=new SettingsReader(appset.outFolderPath+"\\view1tmpF.s",appset.outFolderPath+"\\view1.s");
                        setF.setDataFilePath(appset.outFolderPath+"\\Jinputnew.arff");
                     }
                     else{
                         setF=new SettingsReader(appset.outFolderPath+"/view1tmpF.s",appset.outFolderPath+"/view1.s");
                         setF.setDataFilePath(appset.outFolderPath+"/Jinputnew.arff");
                     }
                 }

                  int endTmp=0;
             if((z+1)*appset.numTargets>(endIndexRR-oldIndexRR))
                 endTmp=endIndexRR;
             else endTmp=(z+1)*appset.numTargets+oldIndexRR;
                 
             int startIndexRR=oldIndexRR+z*appset.numTargets;
             
                 for(int i=startIndexRR;i<endTmp;i++) //do on the fly when reading rules
                        if(rr.rules.get(i).elements.size()>=appset.minSupport)
                            nARules++;
                set.ModifySettings(nARules,dsc1.schema.getNbAttributes());
                if(appset.numSupplementTrees>0)
                    setF.ModifySettingsF(nARules,dsc1.schema.getNbAttributes(),appset);
          }

            //create and modify settings for cicle 2
        if(leftSide1==1 && (endIndexRR1-oldIndexRR1)>z*appset.numTargets){
            if(appset.system.equals("windows")){
                set1=new SettingsReader(appset.outFolderPath+"\\view2tmp1.s",appset.outFolderPath+"\\view2.s");
                set1.setDataFilePath(appset.outFolderPath+"\\Jinputnew1.arff");
            }
            else{
                set1=new SettingsReader(appset.outFolderPath+"/view2tmp1.s",appset.outFolderPath+"/view2.s");
                set1.setDataFilePath(appset.outFolderPath+"/Jinputnew1.arff");
            }
            if(appset.numSupplementTrees>0){
                if(appset.system.equals("windows")){
                     setF1=new SettingsReader(appset.outFolderPath+"\\view2tmpF1.s",appset.outFolderPath+"\\view2.s");
                     setF1.setDataFilePath(appset.outFolderPath+"\\Jinputnew1.arff");
                }
                else{
                     setF1=new SettingsReader(appset.outFolderPath+"/view2tmpF1.s",appset.outFolderPath+"/view2.s");
                     setF1.setDataFilePath(appset.outFolderPath+"/Jinputnew1.arff");
                }
            }
            
             int endTmp=0;
             if((z+1)*appset.numTargets>(endIndexRR1-oldIndexRR1))
                 endTmp=endIndexRR1;
             else endTmp=oldIndexRR1+(z+1)*appset.numTargets;
             
             int startIndexRR1=oldIndexRR1+z*appset.numTargets;
             
             for(int i=startIndexRR1;i<endTmp;i++)
                  if(rr1.rules.get(i).elements.size()>=appset.minSupport)
                       nARules1++;
             set1.ModifySettings(nARules1,dsc.schema.getNbAttributes());
             if(appset.numSupplementTrees>0)
                setF1.ModifySettingsF(nARules1,dsc.schema.getNbAttributes(),appset);
          }
          else if(rightSide1==1 && (endIndexRR-oldIndexRR)>z*appset.numTargets){

            if(appset.system.equals("windows")){  
              set1=new SettingsReader(appset.outFolderPath+"\\view1tmp1.s",appset.outFolderPath+"\\view1.s");
              set1.setDataFilePath(appset.outFolderPath+"\\Jinputnew1.arff");
            }
            else{
               set1=new SettingsReader(appset.outFolderPath+"/view1tmp1.s",appset.outFolderPath+"/view1.s");
               set1.setDataFilePath(appset.outFolderPath+"/Jinputnew1.arff"); 
            }
              if(appset.numSupplementTrees>0){
                  if(appset.system.equals("windows")){ 
                        setF1=new SettingsReader(appset.outFolderPath+"\\view1tmpF1.s",appset.outFolderPath+"\\view1.s");
                        setF1.setDataFilePath(appset.outFolderPath+"\\Jinputnew1.arff");
                  }
                  else{
                        setF1=new SettingsReader(appset.outFolderPath+"/view1tmpF1.s",appset.outFolderPath+"/view1.s");
                        setF1.setDataFilePath(appset.outFolderPath+"/Jinputnew1.arff"); 
                  }
              }

              int endTmp=0;
             if((z+1)*appset.numTargets>(endIndexRR-oldIndexRR))
                 endTmp=endIndexRR;
             else endTmp=(z+1)*appset.numTargets+oldIndexRR;
              
             int startIndexRR=oldIndexRR+z*appset.numTargets;
             
              for(int i=startIndexRR;i<endTmp;i++)
                  if(rr.rules.get(i).elements.size()>=appset.minSupport)
                        nARules1++;
              set1.ModifySettings(nARules1,dsc1.schema.getNbAttributes());
              if(appset.numSupplementTrees>0)
                    setF1.ModifySettingsF(nARules1,dsc1.schema.getNbAttributes(),appset);
          }

        RuleReader ItRules=new RuleReader();
        RuleReader ItRules1=new RuleReader();
        RuleReader ItRulesF=new RuleReader();
        RuleReader ItRulesF1=new RuleReader();

       //modify dataset for cicle 1
       if(leftSide==1 && (endIndexRR1-oldIndexRR1)>z*appset.numTargets ){
           int startIndexRR1=oldIndexRR1+z*appset.numTargets;
           int endTmp=0;
             if((z+1)*appset.numTargets>(endIndexRR1-oldIndexRR1))
                 endTmp=endIndexRR1;
             else endTmp=(z+1)*appset.numTargets+oldIndexRR1;
        try{
         if(appset.treeTypes.get(1)==1/*appset.typeOfRSTrees==1*/) 
             if(appset.system.equals("windows")){ 
                 dsc.modifyDatasetS(startIndexRR1,endTmp,rr1,appset.outFolderPath+"\\Jinputnew.arff",fidFull,appset);
             }
             else{
                dsc.modifyDatasetS(startIndexRR1,endTmp,rr1,appset.outFolderPath+"/Jinputnew.arff",fidFull,appset); 
             }
         else if(appset.treeTypes.get(1)==0/*appset.typeOfRSTrees==0*/)
             if(appset.system.equals("windows")){ 
                 dsc.modifyDatasetCat(startIndexRR1,endTmp,rr1,appset.outFolderPath+"\\Jinputnew.arff",fidFull,appset);
             }
             else{
                dsc.modifyDatasetCat(startIndexRR1,endTmp,rr1,appset.outFolderPath+"/Jinputnew.arff",fidFull,appset); 
             }
        }
        catch(IOException e){
            e.printStackTrace();
        }
       }
       else if(rightSide==1 && (endIndexRR-oldIndexRR)>z*appset.numTargets){
           int endTmp=0;
             if((z+1)*appset.numTargets>(endIndexRR-oldIndexRR))
                 endTmp=endIndexRR;
             else endTmp=(z+1)*appset.numTargets+oldIndexRR;
              
             int startIndexRR=oldIndexRR+z*appset.numTargets;
             
         try{
             if(appset.treeTypes.get(0)==1/*appset.typeOfLSTrees==1*/)
                 if(appset.system.equals("windows")){ 
                    dsc.modifyDatasetS(startIndexRR,endTmp,rr,appset.outFolderPath+"\\Jinputnew.arff",fidFull,appset);
                 }
                 else{
                     dsc.modifyDatasetS(startIndexRR,endTmp,rr,appset.outFolderPath+"/Jinputnew.arff",fidFull,appset); 
                 }
             else if(appset.treeTypes.get(0)==0/*appset.typeOfLSTrees==0*/)
                  if(appset.system.equals("windows")){ 
                    dsc.modifyDatasetCat(startIndexRR,endTmp ,rr,appset.outFolderPath+"\\Jinputnew.arff",fidFull,appset);
                  }
                  else{
                     dsc.modifyDatasetCat(startIndexRR,endTmp ,rr,appset.outFolderPath+"/Jinputnew.arff",fidFull,appset); 
                  }
        }
        catch(IOException e){
            e.printStackTrace();
        }  
       }

       //modify dataset for cicle 2
       if(leftSide1==1 && (endIndexRR1-oldIndexRR1)>z*appset.numTargets){
           int startIndexRR1=oldIndexRR1+z*appset.numTargets;
           int endTmp=0;
             if((z+1)*appset.numTargets>(endIndexRR1-oldIndexRR1))
                 endTmp=endIndexRR1;
             else endTmp=(z+1)*appset.numTargets+oldIndexRR1;
             
        try{
            if(appset.treeTypes.get(1)==1)
                if(appset.system.equals("windows")){ 
                    dsc1.modifyDatasetS(startIndexRR1,endTmp, rr1,appset.outFolderPath+"\\Jinputnew1.arff",fidFull,appset);
                }
                else{
                   dsc1.modifyDatasetS(startIndexRR1,endTmp, rr1,appset.outFolderPath+"/Jinputnew1.arff",fidFull,appset); 
                }
            else if(appset.treeTypes.get(1)==0/*appset.typeOfRSTrees==0*/)
                if(appset.treeTypes.get(1)==1)
                     dsc1.modifyDatasetCat(startIndexRR1,endTmp, rr1,appset.outFolderPath+"\\Jinputnew1.arff",fidFull,appset);
                else
                     dsc1.modifyDatasetCat(startIndexRR1,endTmp, rr1,appset.outFolderPath+"/Jinputnew1.arff",fidFull,appset);
        }
        catch(IOException e){
            e.printStackTrace();
        }
       }
       else if(rightSide1==1 && (endIndexRR-oldIndexRR)>z*appset.numTargets){
           int endTmp=0;
             if((z+1)*appset.numTargets>(endIndexRR-oldIndexRR))
                 endTmp=endIndexRR;
             else endTmp=(z+1)*appset.numTargets+oldIndexRR;
              
             int startIndexRR=oldIndexRR+z*appset.numTargets;
             
         try{
             if(appset.treeTypes.get(0)==1/*appset.typeOfLSTrees==1*/)
                 if(appset.system.equals("windows"))
                    dsc1.modifyDatasetS(startIndexRR,endTmp,rr,appset.outFolderPath+"\\Jinputnew1.arff",fidFull,appset);
                 else
                    dsc1.modifyDatasetS(startIndexRR,endTmp,rr,appset.outFolderPath+"/Jinputnew1.arff",fidFull,appset); 
             else if(appset.treeTypes.get(0)==0/*appset.typeOfRSTrees==0*/)
                 if(appset.system.equals("windows"))
                 dsc1.modifyDatasetCat(startIndexRR,endTmp,rr,appset.outFolderPath+"\\Jinputnew1.arff",fidFull,appset);
             else
                   dsc1.modifyDatasetCat(startIndexRR,endTmp,rr,appset.outFolderPath+"/Jinputnew1.arff",fidFull,appset);   
        }
        catch(IOException e){
            e.printStackTrace();
        }
       }
       
       dsc=null;
       dsc1=null;
      
       if((appset.useNC.get(0)==true && rightSide==1 && appset.networkInit==false && appset.useNetworkAsBackground==false) || (appset.useNC.get(1)==true && leftSide==1 && appset.networkInit==false && appset.useNetworkAsBackground==false)){ 
           if(appset.system.equals("windows")){  
             nclMat.resetFile(new File(appset.outFolderPath+"\\distances.csv"));
             nclMat.writeToFile(new File(appset.outFolderPath+"\\distances.csv"), fidFull,appset);
            }
           else{
              nclMat.resetFile(new File(appset.outFolderPath+"/distances.csv"));
             nclMat.writeToFile(new File(appset.outFolderPath+"/distances.csv"), fidFull,appset); 
           }
       }

         if(leftSide==1 && (endIndexRR1-oldIndexRR1)>z*appset.numTargets){
             exec.run(appset.javaPath, appset.clusPath, appset.outFolderPath, "view2tmp.s"/*"wbtmp.s"*/, 0,appset.clusteringMemory);//was 1 for rules before
             if(appset.numSupplementTrees>0)
                exec.run(appset.javaPath, appset.clusPath, appset.outFolderPath, "view2tmpF.s"/*"wbtmp.s"*/, 0,appset.clusteringMemory);//was 1 for rules before
         }
         else if(rightSide==1 && (endIndexRR-oldIndexRR)>z*appset.numTargets){
             exec.run(appset.javaPath, appset.clusPath, appset.outFolderPath, "view1tmp.s"/*"unctadtmp.s"*/, 0,appset.clusteringMemory);
             if(appset.numSupplementTrees>0)
                 exec.run(appset.javaPath, appset.clusPath, appset.outFolderPath, "view1tmpF.s"/*"unctadtmp.s"*/, 0,appset.clusteringMemory);
         }

        if((appset.useNC.get(0)==true && rightSide1==1 && appset.networkInit==false && appset.useNetworkAsBackground==false) || (appset.useNC.get(1)==true && leftSide1==1 && appset.networkInit==false && appset.useNetworkAsBackground==false)){ 
              if(appset.system.equals("windows")){ 
                 nclMat1.resetFile(new File(appset.outFolderPath+"\\distances.csv"));
                 nclMat1.writeToFile(new File(appset.outFolderPath+"\\distances.csv"), fidFull,appset);
              }
              else{
                 nclMat1.resetFile(new File(appset.outFolderPath+"/distances.csv"));
                 nclMat1.writeToFile(new File(appset.outFolderPath+"/distances.csv"), fidFull,appset);
              }
        }

         if(leftSide1==1 && (endIndexRR1-oldIndexRR1)>z*appset.numTargets){
             exec.run(appset.javaPath, appset.clusPath, appset.outFolderPath,"view2tmp1.s" /*"wbtmpS1.s"*/, 0,appset.clusteringMemory);
             if(appset.numSupplementTrees>0)
                exec.run(appset.javaPath, appset.clusPath, appset.outFolderPath,"view2tmpF1.s" /*"wbtmpS1.s"*/, 0,appset.clusteringMemory);
         }
         else if(rightSide1==1 && (endIndexRR-oldIndexRR)>z*appset.numTargets){
             exec.run(appset.javaPath, appset.clusPath, appset.outFolderPath,"view1tmp1.s"/*"unctadtmpS1.s"*/ , 0,appset.clusteringMemory);
             if(appset.numSupplementTrees>0)
             exec.run(appset.javaPath, appset.clusPath, appset.outFolderPath,"view1tmpF1.s"/*"unctadtmpS1.s"*/ , 0,appset.clusteringMemory);
         }

        String input="", inputF="";
        if(leftSide==1 && (endIndexRR1-oldIndexRR1)>z*appset.numTargets){
            if(appset.system.equals("windows")){ 
                 input=appset.outFolderPath+"\\view2tmp.out";
                 inputF=appset.outFolderPath+"\\view2tmpF.out";
            }
            else{
               input=appset.outFolderPath+"/view2tmp.out";
                 inputF=appset.outFolderPath+"/view2tmpF.out"; 
            }
        }
        else if(rightSide==1 && (endIndexRR-oldIndexRR)>z*appset.numTargets){
            if(appset.system.equals("windows")){ 
                 input=appset.outFolderPath+"\\view1tmp.out";
                    if(appset.numSupplementTrees>0) 
                        inputF=appset.outFolderPath+"\\view1tmpF.out";
            }
            else{
                input=appset.outFolderPath+"/view1tmp.out";
                    if(appset.numSupplementTrees>0) 
                        inputF=appset.outFolderPath+"/view1tmpF.out";
            }
        }
       int newRules=0;
       if((leftSide==1 && (endIndexRR1-oldIndexRR1)>z*appset.numTargets) || (rightSide==1 && (endIndexRR-oldIndexRR)>z*appset.numTargets)){
            ItRules.extractRules(input,fidFull,datJFull,appset);
            ItRules.setSize();
            if(appset.numSupplementTrees>0){
                ItRulesF.extractRules(inputF,fidFull,datJFull,appset);
                ItRulesF.setSize();
            }
       }
        if(leftSide==1 && (endIndexRR1-oldIndexRR1)>z*appset.numTargets){
            if(z==0)
                newRules=rr.addnewRulesC(ItRules, appset.numnewRAttr,1);
            else
                newRules=rr.addnewRulesC(ItRules, appset.numnewRAttr,0);
            if(appset.numSupplementTrees>0)
                rr.addnewRulesCF(ItRulesF, appset.numnewRAttr); 
        }
        else if(rightSide==1 && (endIndexRR-oldIndexRR)>z*appset.numTargets){
            if(z==0)
                newRules=rr1.addnewRulesC(ItRules, appset.numnewRAttr,1);
            else
                newRules=rr1.addnewRulesC(ItRules, appset.numnewRAttr, 0);
            if(appset.numSupplementTrees>0)
                rr1.addnewRulesCF(ItRulesF, appset.numnewRAttr); 
        }     
         //extract rules for cicle 2
        input=""; inputF="";
        if(leftSide1==1 && (endIndexRR1-oldIndexRR1)>z*appset.numTargets){
            if(appset.system.equals("windows")){ 
                    input=appset.outFolderPath+"\\view2tmp1.out";
                    inputF=appset.outFolderPath+"\\view2tmpF1.out";
            }
            else{
                    input=appset.outFolderPath+"/view2tmp1.out";
                    inputF=appset.outFolderPath+"/view2tmpF1.out"; 
            }
        }
        else if(rightSide1==1 && (endIndexRR-oldIndexRR)>z*appset.numTargets){
            if(appset.system.equals("windows")){ 
                    input=appset.outFolderPath+"\\view1tmp1.out";
                    inputF=appset.outFolderPath+"\\view1tmpF1.out";
            }
            else{
                input=appset.outFolderPath+"/view1tmp1.out";
                inputF=appset.outFolderPath+"/view1tmpF1.out";
            }
        }
       int newRules1=0;
       if((leftSide1==1 && (endIndexRR1-oldIndexRR1)>z*appset.numTargets) || (rightSide1==1 && (endIndexRR-oldIndexRR)>z*appset.numTargets)){
        ItRules1.extractRules(input,fidFull,datJFull,appset);
        ItRules1.setSize();
        if(appset.numSupplementTrees>0){
             ItRulesF1.extractRules(inputF,fidFull,datJFull,appset);
             ItRulesF1.setSize();
        }
       }
        if(leftSide1==1 && (endIndexRR1-oldIndexRR1)>z*appset.numTargets){
            if(z==0)
                    newRules1=rr.addnewRulesC(ItRules1, appset.numnewRAttr,1);
            else
                    newRules1=rr.addnewRulesC(ItRules1, appset.numnewRAttr, 0);
            if(appset.numSupplementTrees>0)
                rr.addnewRulesCF(ItRulesF1, appset.numnewRAttr); 
        }
        else if(rightSide1==1 && (endIndexRR-oldIndexRR)>z*appset.numTargets){
            if(z==0)
                    newRules1=rr1.addnewRulesC(ItRules1, appset.numnewRAttr,1);
            else
                    newRules1=rr1.addnewRulesC(ItRules1, appset.numnewRAttr,0);
            if(appset.numSupplementTrees>0)
             rr1.addnewRulesCF(ItRulesF1, appset.numnewRAttr);
        }
       }
       
       if(appset.optimizationType==0){ 
        if(appset.useJoin){
            //add computation of rule support if bagging
            newRedescriptions=rs.createGuidedJoinBasic(rr1, rr, jsN, appset, oldIndexRR1, oldIndexRR, RunInd,oom,fidFull,datJFull, elemFreq, attrFreq, redScores,redScoresAtt,redDistCoverage,redDistCoverageAt, redDistNetwork, targetAtScore, Statistics, maxDiffScoreDistribution,nclMatInit,0);
            if(appset.numSupplementTrees>0){
                 rr.removeRulesCF();
                 rr1.removeRulesCF();
            }
        }
        else if(!appset.useJoin){
            newRedescriptions=rs.createGuidedNoJoinBasic(rr1, rr, jsN, appset, oldIndexRR1, oldIndexRR, RunInd,oom,fidFull,datJFull, elemFreq, attrFreq, redScores,redScoresAtt,redDistCoverage,redDistCoverageAt, redDistNetwork, targetAtScore, Statistics, maxDiffScoreDistribution,nclMatInit,0);
            rr.removeElements(rr.newRuleIndex);
            rr1.removeElements(rr1.newRuleIndex);
            if(appset.numSupplementTrees>0){
                rr.removeRulesCF();
                rr1.removeRulesCF();
            }
        }
       }
       else{
           if(appset.useJoin){
            newRedescriptions=rs.createGuidedJoinExt(rr1, rr, jsN, appset, oldIndexRR1, oldIndexRR, RunInd, oom,fidFull,datJFull);//sqitch sides of rules
        }
        else if(!appset.useJoin){
            newRedescriptions=rs.createGuidedNoJoinExt(rr1, rr, jsN, appset, oldIndexRR1, oldIndexRR, RunInd,oom,fidFull,datJFull);
            rr.removeElements(rr.newRuleIndex);
            rr1.removeElements(rr1.newRuleIndex);
        }
       }
       
         it++;

        for(int nws=2;nws<datJFull.W2indexs.size()+1;nws++){
            if(readers.size()<(datJFull.W2indexs.size()-2)+1)
            readers.add(new RuleReader());
            int oldIndW=readers.get(nws-2).newRuleIndex, endIndW=0;
            
            SettingsReader setMW=new SettingsReader();//(appset.outFolderPath+"\\view3tmp.s",appset.outFolderPath+"\\view2.s");
           if(appset.system.equals("windows")){ 
            setMW.setPath(appset.outFolderPath+"\\view3tmp.s");
            setMW.setStaticFilePath=appset.outFolderPath+"\\view3tmp.s";
            setMW.setDataFilePath(appset.outFolderPath+"\\Jinputnew.arff");
           }
           else{
              setMW.setPath(appset.outFolderPath+"/view3tmp.s");
              setMW.setStaticFilePath=appset.outFolderPath+"/view3tmp.s";
              setMW.setDataFilePath(appset.outFolderPath+"/Jinputnew.arff"); 
           }
            if((nws-1)<(datJFull.W2indexs.size()-2+1))
                setMW.createInitialSettingsGen(nws, datJFull.W2indexs.get(nws-1)+1 ,datJFull.W2indexs.get(nws)+1,datJFull.schema.getNbAttributes() , appset,0);
            else
                setMW.createInitialSettingsGen(nws, datJFull.W2indexs.get(nws-1)+1 ,datJFull.schema.getNbAttributes()+1,datJFull.schema.getNbAttributes() , appset,0);
            
              numBins=0;
        Size=rs.redescriptions.size();
        if(Size%appset.numTargets==0)
            numBins=Size/appset.numTargets;
        else numBins=Size/appset.numTargets+1;
            
        for(int z=0;z<numBins/*percentage.length-1*/;z++){
       
            if(z==0){//should create network from redescriptions!
               if(appset.useNC.size()>nws && appset.networkInit==false){ 
                if(appset.useNC.get(nws)==true && readers.get(nws-2).rules.size()>0){
                    nclMat.reset(appset);
                    if(appset.distanceFilePaths.size()>=nws && appset.networkInit==false && appset.useNetworkAsBackground==false){
                             nclMat.loadDistance(new File(appset.distanceFilePaths.get(nws)), fidFull);
                              if(appset.system.equals("windows")){ 
                                     nclMat.writeToFile(new File(appset.outFolderPath+"\\distance.csv"), fidFull,appset);
                              }
                              else{
                                  nclMat.writeToFile(new File(appset.outFolderPath+"/distance.csv"), fidFull,appset);
                              }
                    }
                     else if(appset.computeDMfromRules==true){
                             nclMat.computeDistanceMatrix(rs.redescriptions, fidFull, appset.maxDistance, datJFull.numExamples,oldRIndex);
                             if(appset.system.equals("windows")){ 
                                    nclMat.resetFile(new File(appset.outFolderPath+"\\distances.csv"));
                                    nclMat.writeToFile(new File(appset.outFolderPath+"\\distances.csv"), fidFull,appset);
                             }
                             else{
                                 nclMat.resetFile(new File(appset.outFolderPath+"/distances.csv"));
                                 nclMat.writeToFile(new File(appset.outFolderPath+"/distances.csv"), fidFull,appset);
                             }
                     }
                   }
               }
                endIndW=readers.get(nws-2).rules.size();
            }
            
            nARules=0; nARules1=0;
            double startPerc=0;//percentage[z];
            double endPerc=0;//percentage[z+1];
            int minCovElements[]=new int[]{0};
            int maxCovElements[]=new int[]{0};
            int cuttof=0;

           if(appset.system.equals("windows")) 
                dsc=new DataSetCreator(appset.outFolderPath+"\\JinputAll.arff");
           else
               dsc=new DataSetCreator(appset.outFolderPath+"/JinputAll.arff");
            
             try{
        dsc.readDataset();
        }
        catch(IOException e){
            e.printStackTrace();
        }
            
            int endTmp=0;
             if((z+1)*appset.numTargets>rs.redescriptions.size())
                 endTmp=rs.redescriptions.size();
             else endTmp=(z+1)*appset.numTargets;
              //add conditions in the this part of the code...!!!
             int startIndexRR=oldRIndex[0]+z*appset.numTargets;
            
            for(int i=startIndexRR;i<endTmp;i++)//oldRIndex[0];i<rs.redescriptions.size();i++) //do on the fly when reading rules
                        nARules++;
             setMW.ModifySettings(nARules,dsc.schema.getNbAttributes());
             try{
                 if(appset.treeTypes.get(nws)==1/*appset.typeOfRSTrees==1*/){ 
                     if(appset.system.equals("windows")) 
                         dsc.modifyDatasetS(startIndexRR,endTmp, rs.redescriptions,appset.outFolderPath+"\\Jinputnew.arff",fidFull,appset);
                     else 
                         dsc.modifyDatasetS(startIndexRR,endTmp, rs.redescriptions,appset.outFolderPath+"/Jinputnew.arff",fidFull,appset);
                 }
         else if(appset.treeTypes.get(nws)==0/*appset.typeOfRSTrees==0*/){
             if(appset.system.equals("windows")) 
                dsc.modifyDatasetCat(startIndexRR,endTmp, rs.redescriptions,appset.outFolderPath+"\\Jinputnew.arff",fidFull,appset);
             else
                dsc.modifyDatasetCat(startIndexRR,endTmp, rs.redescriptions,appset.outFolderPath+"/Jinputnew.arff",fidFull,appset); 
         }
        }
        catch(IOException e){
            e.printStackTrace();
        }
             
             exec.run(appset.javaPath, appset.clusPath, appset.outFolderPath, "view3tmp.s"/*"wbtmp.s"*/, 0,appset.clusteringMemory);//was 1 for rules before
             
             String input;
             if(appset.system.equals("windows")) 
              input=appset.outFolderPath+"\\view3tmp.out";
             else
                 input=appset.outFolderPath+"/view3tmp.out";
             
              int newRules=0;
              RuleReader ItRules=new RuleReader();
              
             ItRules.extractRules(input,fidFull,datJFull,appset);
        ItRules.setSize();
            if(z==0)
                newRules=readers.get(nws-2).addnewRulesC(ItRules, appset.numnewRAttr,1);
            else
                newRules=readers.get(nws-2).addnewRulesC(ItRules, appset.numnewRAttr,0);
           }
        
        if(appset.useJoin){//do redescription construction
            rs.combineViewRulesJoin(readers.get(nws-2), jsN, appset, oldIndW, RunInd, oom, fidFull, datJFull, oldRIndex ,nws); 
        }
        else{//(rr, rr1, jsN, appset, oldIndexRR, oldIndexRR1, RunInd, oom,fid,datJ);
            rs.combineViewRules(readers.get(nws-2), jsN, appset, oldIndW, RunInd, oom, fidFull, datJFull, oldRIndex ,nws);
        }
           
        //rs.combineViewRules(readers.get(nws-2), jsN, appset, oldIndW, RunInd, oom, fid, datJ, oldRIndex ,nws);
        }
        
        if(leftSide==1){
            leftSide=0;
            rightSide=1;
        }
        else if(rightSide==1){
            rightSide=0;
            leftSide=1;
        }
        if(leftSide1==1){
            leftSide1=0;
            rightSide1=1;
        }
        else if(rightSide1==1){
            rightSide1=0;
            leftSide1=1;
        }
        RunInd++;
        }
      }
         
        //removing all redescriptions with inadequate minSupport and minJS
        rs.remove(appset);
       
      int numFullRed=0;
        //computing pVal...
        numFullRed=rs.computePVal(datJFull,fidFull);
        rs.removePVal(appset);
        
         int minimize=0;
         if(appset.minimizeRules==true)
             minimize=1;
        
         rs.adaptSet(datJFull, fidFull,0);
         rs.sortRedescriptions();
       
        if(appset.sameView==1)
        for(int k=rs.redescriptions.size()-1;k>=0;k--)
            if(rs.redescriptions.get(k).disj(datJFull, fidFull) == 0)
                rs.redescriptions.remove(k);
        
         for(int i=0;i<rs.redescriptions.size();i++){

             if(appset.classHomogeneity == true)
             rs.redescriptions.get(i).setClassCount(targets, targetsToIndex, fidFull);
         }
         
          if(appset.useSplitTesting==true){
              for(int i=0;i<rs.redescriptions.size();i++){
                    // rs.redescriptions.get(i).ComputeValidationStatistics(datJFull,datJTrainValid, fidFull);//ovdje koristiti train+validation???
                    rs.redescriptions.get(i).ComputeValidationStatistics(datJFull,datJFull, fidFull);
                     rs.redescriptions.get(i).ComputeTestStatistics(datJFull,datJTest, fidFull);
              }
              
              for(int k=rs.redescriptions.size()-1;k>=0;k--){
                  if(rs.redescriptions.get(k).JSTest==0)
                      rs.redescriptions.remove(k);
              }
              
          }
         
         
         for(int i=0;i<rs.redescriptions.size();i++)
             rs.redescriptions.get(i).clearRuleMaps();
         
        if(appset.attributeImportance==0) 
         rs.adaptSet(datJFull, fidFull,minimize);
        else
            rs.adaptSet(datJFull, fidFull, 0);

      RedescriptionSet Result=rs;
      
      if(appset.optimizationType==0){
      
      double sumN=0.0;
      double heuristicWeights[]=appset.preferences.get(0);
      double coverage[]=new double[2];
      double ResultsScore=Result.computeRedescriptionSetScore(heuristicWeights,coverage,datJFull,fidFull);
       
     if(appset.system.equals("windows")){ 
      Result.writeToFile(appset.outFolderPath+"\\"+appset.outputName+(1)+".rr", datJFull, fidFull, startTime,numFullRed,appset, ResultsScore, coverage,oom);
      Result.writePlots(appset.outFolderPath+"\\"+"RuleData"+(1)+".csv", appset,datJFull,fidFull);
     }
     else{
       Result.writeToFile(appset.outFolderPath+"/"+appset.outputName+(1)+".rr", datJFull, fidFull, startTime,numFullRed,appset, ResultsScore, coverage,oom);  
        Result.writePlots(appset.outFolderPath+"/"+"RuleData"+(1)+".csv", appset,datJFull,fidFull);
     }
      }
      else{
          double coverage[];
           rs.computeAllMeasureFS(datJFull, appset, fidFull);
           
           for(int i=rs.redescriptions.size()-1;i>=0;i--)
                if(rs.redescriptions.get(i).elementsTest.size()<5)
                    rs.redescriptions.remove(i);
         
      double ResultsScore=0.0;
      
         CoocurenceMatrix coc=null;
         
         if(datJFull.numExamples<10000 && datJFull.schema.getNbAttributes()-1<10000){
                coc=new CoocurenceMatrix(datJFull.numExamples,datJFull.schema.getNbAttributes()-1);
                coc.computeMatrix(rs, datJFull); 
         }
         
       Result=new RedescriptionSet();
      
      double sumN=0.0;
      
      if(appset.parameters.size()==0 && appset.exhaustiveTesting==0){
          ArrayList<Double> par=new ArrayList<>();
          par.add(appset.minJS); par.add((double)appset.minSupport);  par.add((double)appset.missingValueJSType);
          System.out.println("Configuring the default parameters...");
          appset.parameters.add(par);
      }
     
      if(appset.exhaustiveTesting==0){
          RedescriptionSet predFinal = null;
      for(int i=0;i<appset.parameters.size();i++){
          appset.minJS=appset.parameters.get(i).get(0);
          appset.minSupport= appset.parameters.get(i).get(1).intValue();
           Result=new RedescriptionSet();

      ArrayList<RedescriptionSet> resSets=null; 
      if(datJFull.numExamples<10000 && datJFull.schema.getNbAttributes()-1<10000)
          if(appset.classHomogeneity == true)
            resSets=Result.createRedescriptionSetsCoocGenPred(rs,appset.preferences,appset.parameters.get(i).get(2).intValue(), appset,datJFull,fidFull,coc);//adds the most specific redescription first
          else  resSets=Result.createRedescriptionSetsCoocGen(rs,appset.preferences,appset.parameters.get(i).get(2).intValue(), appset,datJFull,fidFull,coc);//adds the most specific redescription first
      else
          resSets=Result.createRedescriptionSetsRandGen(rs,appset.preferences,appset.parameters.get(i).get(2).intValue(), appset,datJFull,fidFull,coc);//should add one highly accurate redescription at random
      
      if(resSets==null){//perhaps create a null file
          break;
      }

      for(int rset=0;rset<resSets.size();rset++)
            resSets.get(rset).computeLift(datJFull, fidFull);
  
      System.out.println("exhaustiveTesting = 0");
      System.out.println("RS size: "+resSets.size());
      
     for(int fit=0;fit<resSets.size();fit++){
       coverage=new double[2];

      ResultsScore=resSets.get(fit).computeRedescriptionSetScoreGen(appset.preferences.get(fit),appset.parameters.get(i).get(2).intValue(),coverage,datJFull,appset,fidFull);
      System.out.println("Results score: "+ResultsScore);
      numFullRed=resSets.get(fit).computePVal(datJFull,fidFull);

      if(appset.system.equals("windows"))
         resSets.get(fit).writeToFile(appset.outFolderPath+"\\"+appset.outputName+"StLev_"+fit+" minjs "+appset.minJS+" JSType "+appset.parameters.get(i).get(2).intValue()+".rr", datJFull, fidFull, startTime,numFullRed,appset, ResultsScore, coverage,oom);
      else
        resSets.get(fit).writeToFile(appset.outFolderPath+"/"+appset.outputName+"StLev_"+fit+" minjs "+appset.minJS+" JSType "+appset.parameters.get(i).get(2).intValue()+".rr", datJFull, fidFull, startTime,numFullRed,appset, ResultsScore, coverage,oom);  
      
      if(appset.system.equals("windows"))
             resSets.get(fit).writePlots(appset.outFolderPath+"\\"+"RuleData"+"StLev_"+fit+" minjs "+appset.minJS+"JSType "+appset.parameters.get(i).get(2).intValue()+".csv", appset,datJFull,fidFull);
      else
            resSets.get(fit).writePlots(appset.outFolderPath+"/"+"RuleData"+"StLev_"+fit+" minjs "+appset.minJS+"JSType "+appset.parameters.get(i).get(2).intValue()+".csv", appset,datJFull,fidFull);

      predFinal = resSets.get(0);
     }
      }
      
      resultSetOut.redescriptions.addAll(predFinal.redescriptions);
      System.out.println("Num reds final: "+resultSetOut.redescriptions.size());

      RuleReader rrAll = new RuleReader();
      //need to add elements sto rules
      int numTest =0;
      for(int i=0;i<rr.rules.size();i++){
          numTest = 0;
          rr.rules.get(i).closeInterval(datJFull, fidFull);
         // System.out.println(rr.rules.get(i).rule);
          if(appset.classHomogeneity == true)
           rr.rules.get(i).setClassCount(targets, targetsToIndex, fidFull);
          rr.rules.get(i).addElements1(fidFull, fidFull ,datJFull);
          numTest = rr.rules.get(i).numElementsInTest(fidFull, fidFull ,datJTest);
          if(numTest >= 5)
               rrAll.rules.add(rr.rules.get(i));
      }
      
     //rrAll.rules.addAll(rr.rules);

     for(int i=0;i<rr1.rules.size();i++){
         numTest =0;
         rr1.rules.get(i).closeInterval(datJFull, fidFull);
        // System.out.println(rr1.rules.get(i).rule);
         if(appset.classHomogeneity == true)
         rr1.rules.get(i).setClassCount(targets, targetsToIndex, fidFull);
          rr1.rules.get(i).addElements1(fidFull,fidFull ,datJFull);  
          numTest = rr1.rules.get(i).numElementsInTest(fidFull, fidFull ,datJTest);
          if(numTest >=5)
               rrAll.rules.add(rr1.rules.get(i));
      }
    
     rrAll.rules.addAll(rr1.rules);
     
     String ruleFile = appset.outFolderPath;
     
          rrAll.filterRules(0, 0.99, appset);//0.8 default//0.5 last value
          if(appset.classHomogeneity == true)
            rrAll.filterRulesClassPurity(0, 1.0 ,appset);// 0.6 default//0.8 last value
          
          System.out.println("Num descriptive rules final: "+rrAll.rules.size());
     
     if(appset.system.equals("windows"))
         ruleFile+="\\rules.rr";
     else ruleFile+="/rules.rr";
     
     rrResultOut.rules.addAll(rrAll.rules); 
     
     try{
         FileWriter fw = new FileWriter(ruleFile);
         fw.write("Rules: \n\n");
         
         for(int i=0;i<rrAll.rules.size();i++){
             fw.write(rrAll.rules.get(i).rule+"\n");
             
             fw.write("Target value counts: \n");
             for(int j=0;j<rrAll.rules.get(i).targetValueCounts.size();j++)
                 fw.write(rrAll.rules.get(i).targetValueCounts.get(j)+" ");
             TIntIterator it = rrAll.rules.get(i).elements.iterator();
             fw.write("\n\nElements: ");
             while(it.hasNext())
                    fw.write(fidFull.idExample.get(it.next())+" ");
             fw.write("\n\n");
                     }
               
               fw.close();
     }
     catch(IOException e){
         e.printStackTrace();
     }
    }
   }
 }
    
      static void createSupervisedRules(ApplicationSettings appset, Mappings fid, DataSetCreator datJTest, WekaDatasetLoader wdl, ArrayList<Rule> rulesSupervised){
             PART pc = new PART();
        try{
             String[] options = weka.core.Utils.splitOptions("weka.classifiers.rules.PART -C 0.25 -M 10 -num-decimal-places 4");
             pc.setOptions(options);
            Instances TrainValidationP = null;
            
             Cleaner c = new Cleaner();
             String p = "";
             
              if(appset.system.equals("windows"))
                  p = appset.outFolderPath+"\\TrainValidation.arff";
              else p = appset.outFolderPath+"/TrainValidation.arff";
             
               c.clean(p);
            
             if(appset.system.equals("windows"))
                TrainValidationP = wdl.loadDataset(appset.outFolderPath+"\\TrainValidation.arff");
             else TrainValidationP = wdl.loadDataset(appset.outFolderPath+"/TrainValidation.arff");
           
            Remove rm = new Remove();
              rm.setAttributeIndices("first");
              rm.setInputFormat(TrainValidationP);
               TrainValidationP = Filter.useFilter(TrainValidationP, rm);
                TrainValidationP.setClass(TrainValidationP.attribute(TrainValidationP.numAttributes()-1));
               pc.buildClassifier(TrainValidationP);
               String s = pc.toString();
               
               String parts[] = s.split("\n\\s*\n");
               System.out.println("Num parts: "+parts.length);
               FileWriter fw = null;
               if(appset.system.equals("windows"))
                fw = new FileWriter(appset.outFolderPath+"\\supRules.rr");
               else  fw = new FileWriter(appset.outFolderPath+"/supRules.rr");
               String sp="";
               for(int i=2;i< parts.length-2;i++){
                   sp = parts[i];
                   int cl = Integer.parseInt(sp.split(":")[1].split(" ")[1]);
                   sp = sp.split(":")[0];
                   sp = sp.replaceAll("\n", " ");
                   System.out.println(sp+"\n");
                    //fw.write(sp+" : "+cl+"\n");
                    sp = sp.replaceAll("−", "-");
                    fw.write(sp+"\n");
                    rulesSupervised.add(new Rule(sp,fid));
                    rulesSupervised.get(rulesSupervised.size()-1).ConstructRule1(sp, fid);
                    rulesSupervised.get(rulesSupervised.size()-1).addElements(fid, datJTest);
                    System.out.println("Ucitano: "+rulesSupervised.get(rulesSupervised.size()-1).elements.size());
               }
               
               JRip jr = new JRip();
               String options1[] = weka.core.Utils.splitOptions("-F 3 -N 1.0 -O 500 -S 1 -num-decimal-places 4 -batch-size 200");
               jr.setOptions(options1);
               jr.buildClassifier(TrainValidationP);
               
               System.out.println("JRip: ");
               System.out.println(jr);
               String s1 = jr.toString();
               parts = s1.split("\n\\s*\n");
               System.out.println("JRip parts: "+parts.length);
               parts = parts[1].split("\n");
                for(int i=0;i< parts.length-1;i++){
                    int cl = Integer.parseInt(parts[i].split("=>")[1].split(" ")[1].split("=")[1]);
                    parts[i] = parts[i].split("=>")[0];
                    System.out.println("cl: "+cl);
                    parts[i] = parts[i].replaceAll("\\(", "");
                    parts[i] = parts[i].replaceAll("\\)", "");
                   sp = parts[i];
                   sp = sp.split(":")[0];
                   sp = sp.replaceAll("\n", " ");
                   System.out.println(sp+"\n");
                   sp = sp.replaceAll("and", "AND");
                   sp = sp.replaceAll("−", "-");
                    rulesSupervised.add(new Rule(sp,fid));
                    rulesSupervised.get(rulesSupervised.size()-1).ConstructRule1(sp, fid);
                    rulesSupervised.get(rulesSupervised.size()-1).addElements(fid, datJTest);
                    System.out.println("Ucitano: "+rulesSupervised.get(rulesSupervised.size()-1).elements.size());
                    //fw.write(sp+" : "+cl+"\n");
                    fw.write(sp+"\n");
               }
               fw.close();    
               
               System.out.println("Num red rules: "+rulesSupervised.size());  
               
               for(int i=rulesSupervised.size()-1;i>=0;i--)
                    if(rulesSupervised.get(i).elements.size()<5)
                        rulesSupervised.remove(i);
                
               System.out.println("Num predictive rules after filtering: "+rulesSupervised.size());
               
        }
        catch(Exception e){
            e.printStackTrace();
        }      
      }
      
    static void createSubgroups(ApplicationSettings appset, Mappings fid, DataSetCreator datJTest, WekaDatasetLoader wdl,ArrayList<Rule> rulesSubgroups, String trainValidationPath, String testPath){
            
		int maxIterations = 8;
		int beamSize = 5;
		// critical value from chi^2 tables, 9.24 corresponds to 2 degrees of freedom and 99% significance
		float minSignificance = 9.24f;
		// parameter for example weight update function
		float gamma = 0.7f;
		
		// select one of the datasets
		DataSet ds = new DataSet(trainValidationPath,false);
                DataSet dsTest = new DataSet(testPath,false);
		
                CN2SD cn2sd = new CN2SD(ds, beamSize, maxIterations, minSignificance, gamma);
                
                ArrayList<sgd.Rule> rules = cn2sd.run();
			/*for (sgd.Rule r : rules) {
				r.evaluate(dsTest);
			}*/
               Rule rS  = null;
               String ruleString[], sS;
               for(sgd.Rule r: rules){
                   sS = r.toString();
                   ruleString = sS.split("\\n");
                   String rs0 = ruleString[0].trim().replaceAll("−", "-");
                   rS = new Rule(rs0,fid);
                   System.out.println(r);
                   rulesSubgroups.add(rS);
                   rulesSubgroups.get(rulesSubgroups.size()-1).ConstructRule1(rs0, fid);
                    rulesSubgroups.get(rulesSubgroups.size()-1).addElements(fid, datJTest);
                    System.out.println("Ucitano: "+rulesSubgroups.get(rulesSubgroups.size()-1).elements.size());
               }
               
               System.out.println("Num rules subgroups: "+rulesSubgroups.size());
               for(int i=rulesSubgroups.size()-1;i>=0;i--)
                    if(rulesSubgroups.get(i).elements.size()<5)
                        rulesSubgroups.remove(i);
                
               System.out.println("Num rules subgroups after filtering: "+rulesSubgroups.size());
               
                FileWriter fw = null;
                try{
               if(appset.system.equals("windows"))
                fw = new FileWriter(appset.outFolderPath+"\\suubgroups.rr");
               else  fw = new FileWriter(appset.outFolderPath+"/subgroups.rr");
               
               
                for(int i=0;i<rulesSubgroups.size();i++)
                    fw.write(rulesSubgroups.get(i).rule+"\n");
               
                    fw.close();
                }
                catch(IOException e){
                    e.printStackTrace();
                }
    }
    
    
    //create TrainValidationAll (for boruta and svetnik, TestAll for boruta), TrainValidationAllSelected, TestAllSelected
    //function creates just TrainValidationAll and TestAll
    //TrainValidationRules (for svetnik), TainValidadtionRulesSelected, TestRulesSelected
    //TrainValidationRedescriptions (for svetnik), TainValidadtionRedescriptionsSelected, TestRedescriptionsSelected
    //TrainValidationSubgroups (for svetnik), TainValidadtionSubgroupsSelected, TestSubgroupsSelected
    static ArrayList<Instances> createFilesForPredictivityTesting(ApplicationSettings appset, Mappings fidFull,Mappings fid, WekaDatasetLoader wdl, RuleReader rrResultOut,RedescriptionSet resultSetOut, ArrayList<Rule> rulesSupervised, ArrayList<Rule> subgroups, Instances data, DataSetCreator datJ, DataSetCreator datJFull, String inputTrainValidation, String inputTest){
        Instances dataTrainValidation = null , dataTest = null, dataset = null;
        
         if(appset.system.equals("windows")){
                dataTrainValidation = wdl.loadDataset(appset.outFolderPath+"\\TrainValidation.arff");
                dataTrainValidation.setClassIndex(dataTrainValidation.numAttributes()-1);
           }
           else{
                dataTrainValidation = wdl.loadDataset(appset.outFolderPath+"/TrainValidation.arff");
                dataTrainValidation.setClassIndex(dataTrainValidation.numAttributes()-1);//class indeks must be last atribute
           }
         
         if(appset.system.equals("windows")){
                dataTest = wdl.loadDataset(appset.outFolderPath+"\\Test.arff");
                dataTest.setClassIndex(dataTest.numAttributes()-1);
           }
           else{
                dataTest = wdl.loadDataset(appset.outFolderPath+"/Test.arff");
                dataTest.setClassIndex(dataTest.numAttributes()-1);//class indeks must be last atribute
           }
         
           if(appset.system.equals("windows")){
                dataset = wdl.loadDataset(appset.outFolderPath+"\\Test.arff");
                dataTest.setClassIndex(dataTest.numAttributes()-1);
           }
           else{
                dataTest = wdl.loadDataset(appset.outFolderPath+"/Test.arff");
                dataTest.setClassIndex(dataTest.numAttributes()-1);//class indeks must be last atribute
           }
         
         ArrayList<Instances> ret = new ArrayList<>();
         int indO = dataTrainValidation.numAttributes()-1;
          for(int i=0;i<rrResultOut.rules.size();i++){
          dataTest.insertAttributeAt(new Attribute("r"+i), dataTest.numAttributes()-1);
          dataTrainValidation.insertAttributeAt(new Attribute("r"+i), dataTrainValidation.numAttributes()-1);
      }

      for(int i=0;i<resultSetOut.redescriptions.size();i++){
          dataTest.insertAttributeAt(new Attribute("R"+i), dataTest.numAttributes()-1);
          dataTrainValidation.insertAttributeAt(new Attribute("R"+i), dataTrainValidation.numAttributes()-1);
      }

      for(int i=0;i<rulesSupervised.size();i++){
          dataTest.insertAttributeAt(new Attribute("rs"+i), dataTest.numAttributes()-1);
          dataTrainValidation.insertAttributeAt(new Attribute("rs"+i), dataTrainValidation.numAttributes()-1);
      }
      
       for(int i=0;i<subgroups.size();i++){
          dataTest.insertAttributeAt(new Attribute("S"+i), dataTest.numAttributes()-1);
          dataTrainValidation.insertAttributeAt(new Attribute("S"+i), dataTrainValidation.numAttributes()-1);
      }
       
       //compute entities for reds, supervised rules and subgroups on dataset containing all instances

       
       //add instance into data
       String entN1="";
       
       for(int j=0;j<rrResultOut.rules.size();j++) 
      for(int i=0;i<dataTrainValidation.numInstances();i++){
          entN1 = dataTrainValidation.instance(i).stringValue(0);
          if(!fidFull.exampleId.containsKey(entN1)){
              if(fidFull.exampleId.containsKey("\"+entN+\""))
                  entN1 = "\""+entN1+"\"";
          }
          if(rrResultOut.rules.get(j).elements.contains(fidFull.exampleId.get(entN1)))
                    dataTrainValidation.instance(i).setValue(indO+j, 1);
          else  dataTrainValidation.instance(i).setValue(indO+j, 0);
      }
       
       String entN="";
     	 for(int j=0;j<resultSetOut.redescriptions.size();j++){
          for(int i=0;i<dataTrainValidation.numInstances();i++){
          entN = dataTrainValidation.instance(i).stringValue(0);
          if(resultSetOut.redescriptions.get(j).elements.contains(fidFull.exampleId.get(entN)))
                    dataTrainValidation.instance(i).setValue(indO+j+rrResultOut.rules.size(), 1);
          else  dataTrainValidation.instance(i).setValue(indO+j+rrResultOut.rules.size(), 0);
      }
     }  
       
       
       for(int i=0;i<rulesSupervised.size();i++){
           rulesSupervised.get(i).elements.clear();
            rulesSupervised.get(i).addElements(fidFull, datJFull);
       }
     
        for(int j=0;j<rulesSupervised.size();j++) 
      for(int i=0;i<dataTrainValidation.numInstances();i++){
          entN1 = dataTrainValidation.instance(i).stringValue(0);
          if(!fidFull.exampleId.containsKey(entN1)){
              if(fidFull.exampleId.containsKey("\"+entN+\""))
                  entN1 = "\""+entN1+"\"";
          }
          if(rulesSupervised.get(j).elements.contains(fidFull.exampleId.get(entN1)))
                    dataTrainValidation.instance(i).setValue(indO+rrResultOut.rules.size()+resultSetOut.redescriptions.size()+j, 1);
          else  dataTrainValidation.instance(i).setValue(indO+rrResultOut.rules.size()+resultSetOut.redescriptions.size()+j, 0);
      }
        
        
       for(int i=0;i<subgroups.size();i++){
           subgroups.get(i).elements.clear();
            subgroups.get(i).addElements(fidFull, datJFull);
       }
     
        for(int j=0;j<subgroups.size();j++) 
      for(int i=0;i<dataTrainValidation.numInstances();i++){
          entN1 = dataTrainValidation.instance(i).stringValue(0);
          if(!fidFull.exampleId.containsKey(entN1)){
              if(fidFull.exampleId.containsKey("\"+entN+\""))
                  entN1 = "\""+entN1+"\"";
          }
          if(subgroups.get(j).elements.contains(fidFull.exampleId.get(entN1)))
                    dataTrainValidation.instance(i).setValue(indO+rrResultOut.rules.size()+resultSetOut.redescriptions.size()+rulesSupervised.size()+j, 1);
          else  dataTrainValidation.instance(i).setValue(indO+rrResultOut.rules.size()+resultSetOut.redescriptions.size()+rulesSupervised.size()+j, 0);
      }  
     
        
        //test
         for(int j=0;j<rrResultOut.rules.size();j++) 
      for(int i=0;i<dataTest.numInstances();i++){
          entN1 = dataTest.instance(i).stringValue(0);
          if(!fidFull.exampleId.containsKey(entN1)){
              if(fidFull.exampleId.containsKey("\"+entN+\""))
                  entN1 = "\""+entN1+"\"";
          }
          if(rrResultOut.rules.get(j).elements.contains(fidFull.exampleId.get(entN1)))
                    dataTest.instance(i).setValue(indO+j, 1);
          else  dataTest.instance(i).setValue(indO+j, 0);
      }
      
          for(int j=0;j<resultSetOut.redescriptions.size();j++){
          for(int i=0;i<dataTest.numInstances();i++){
          entN = dataTest.instance(i).stringValue(0);
          if(resultSetOut.redescriptions.get(j).elements.contains(fidFull.exampleId.get(entN)))
                    dataTest.instance(i).setValue(indO+j+rrResultOut.rules.size(), 1);
          else  dataTest.instance(i).setValue(indO+j+rrResultOut.rules.size(), 0);
      }
     }  
          
       for(int j=0;j<rulesSupervised.size();j++) 
      for(int i=0;i<dataTest.numInstances();i++){
          entN1 = dataTest.instance(i).stringValue(0);
          if(!fidFull.exampleId.containsKey(entN1)){
              if(fidFull.exampleId.containsKey("\"+entN+\""))
                  entN1 = "\""+entN1+"\"";
          }
          if(rulesSupervised.get(j).elements.contains(fidFull.exampleId.get(entN1)))
                    dataTest.instance(i).setValue(indO+rrResultOut.rules.size()+resultSetOut.redescriptions.size()+j, 1);
          else  dataTest.instance(i).setValue(indO+rrResultOut.rules.size()+resultSetOut.redescriptions.size()+j, 0);
      }
       
      for(int j=0;j<subgroups.size();j++) 
      for(int i=0;i<dataTest.numInstances();i++){
          entN1 = dataTest.instance(i).stringValue(0);
          if(!fidFull.exampleId.containsKey(entN1)){
              if(fidFull.exampleId.containsKey("\"+entN+\""))
                  entN1 = "\""+entN1+"\"";
          }
          if(subgroups.get(j).elements.contains(fidFull.exampleId.get(entN1)))
                    dataTest.instance(i).setValue(indO+rrResultOut.rules.size()+resultSetOut.redescriptions.size()+rulesSupervised.size()+j, 1);
          else  dataTest.instance(i).setValue(indO+rrResultOut.rules.size()+resultSetOut.redescriptions.size()+rulesSupervised.size()+j, 0);
      }  
       
        String outputFilenamePred = "";
        
            if(appset.system.equals("windows"))
                        outputFilenamePred= appset.outFolderPath+"\\TrainValidationAllFeatures.arff";
                    else outputFilenamePred= appset.outFolderPath+"/TrainValidationAllFeatures.arff";
                    try{
                     ConverterUtils.DataSink.write(outputFilenamePred, dataTrainValidation);
                    }
                   catch(Exception e){
                       e.printStackTrace();
                   }
        
                    if(appset.system.equals("windows"))
                        outputFilenamePred= appset.outFolderPath+"\\TestAllFeatures.arff";
                    else outputFilenamePred= appset.outFolderPath+"/TestAllFeatures.arff";
                    try{
                     ConverterUtils.DataSink.write(outputFilenamePred, dataTest);
                    }
                   catch(Exception e){
                       e.printStackTrace();
                   }
       
       ret.add(dataTrainValidation); ret.add(dataTest);
         
         return ret;
        
    }
    
    
    static void runRFunctions(String inputR, String trainValidationInput, String testInput, String datasetIn, String outputPath, String libraryPath){
        Rengine engine = Rengine.getMainEngine();
        if(engine == null){
            engine=new Rengine (new String [] {"--vanilla"}, false, null);
        if (!engine.waitForR())
        {
            System.out.println ("Cannot load R");
            return;
        }
    }
        
        String rFunkcija = inputR;
      
      String input = trainValidationInput;
      String inputTest = testInput;
      String dataset = datasetIn;
      String output = outputPath;
     
      engine.eval("input<-'"+input+"'");
      String s = engine.eval("input").asString();
       System.out.println(s);
      engine.eval("inputTest<-'"+inputTest+"'");
      engine.eval("dataset<-'"+dataset+"'");
      engine.eval("output<-'"+output+"'");
      
      engine.eval("library('Boruta',lib.loc = \""+libraryPath+"\")");
      engine.eval("library('foreign')");
      engine.eval("library('ranger',lib.loc = \""+libraryPath+"\")");
      engine.eval("library('randomForest',lib.loc = \""+libraryPath+"\")");
      engine.eval("library('varSelRF',lib.loc = \""+libraryPath+"\")");
      String ver = engine.eval("version").toString();
      System.out.println("version: "+ver);
      
      engine.eval("source('" +rFunkcija + "')");
      engine.eval("X <- lsf.str()");
      String t2 = engine.eval("as.vector(X)").toString();
      System.out.println("t2: "+t2);
      String pr1 = engine.eval("search()").toString();
      System.out.println("pr1: "+pr1);
      String ss = engine.eval("computeFeaturesAndDatasetsRJS").asString();
      System.out.println(ss);
      /*String r = */engine.eval("computeFeaturesAndDatasetsRJS(input,inputTest,dataset,output)");//.asString();  //run Boruta on test set as well for comparisson
      //System.out.println("r: "+r);  
      engine.end();
    }
    
    public static void trainTestClassifiers(ApplicationSettings appset, WekaDatasetLoader wdl, String dataTrainValidationInput, String dataTestInput, HashMap<String,ArrayList<Double>> resultsWeka){
     
     Instances dataTrainValidation = null;
     Instances dataTest = null;
     
     
     dataTrainValidation = wdl.loadDataset(dataTrainValidationInput);
     dataTest = wdl.loadDataset(dataTestInput);
        
     SettingsReader predictiveSettings = new SettingsReader();
     int numAttributes = dataTrainValidation.numAttributes();
     
      if(appset.system.equals("windows")){//createPredictiveSettingsFunction
             predictiveSettings.setPath(appset.outFolderPath+"\\predictiveFinFC.s");//replace view1 with real name
            predictiveSettings.createPredictiveSettings(dataTrainValidationInput, dataTestInput, numAttributes, 1, 600, 0);

           //  predictiveSettings.setDataFilePath(appset.outFolderPath+"\\JinputInitial.arff");
     }
     else{
             predictiveSettings.setPath(appset.outFolderPath+"/predictiveFinFC.s");
               predictiveSettings.createPredictiveSettings(dataTrainValidationInput, dataTestInput, numAttributes, 1, 600, 0);

     }
     
     //call CLUS
     
     ClusProcessExecutor exec=new ClusProcessExecutor();
       
      exec.run(appset.javaPath,appset.clusPath ,appset.outFolderPath,"predictiveFinFC.s",0, appset.clusteringMemory);
      
       //add all results from PCTRF
       double auc = 0.0, auprc=0.0;
      try{
             Path p= null;
              BufferedReader read = null;
              String line = "";
              
               if(appset.system.equals("windows"))
                p= Paths.get(appset.outFolderPath+"\\predictiveFinFC.out");
              else  p= Paths.get(appset.outFolderPath+"/predictiveFinFC.out");
              
              read = Files.newBufferedReader(p);
              line ="";
              
              int perfLine = 0;
               auc=0; auprc=0;
              
              while((line = read.readLine())!=null){
                  
                  if(line.contains("averageAUROC") || line.contains("averageAUPRC")){
                     if(line.contains("averageAUROC"))
                         perfLine = 1;
                     else perfLine = 2;
                     continue;
                  }
                  
                  if(perfLine == 1){
                      if(line.contains("Default"))
                          continue;
                      else{
                          String tmpT[] = line.split(":");
                          tmpT[1]=tmpT[1].replaceAll("−", "-");
                          auc = Double.parseDouble(tmpT[1].trim());
                      } 
                  }
                  
                  if(perfLine == 2){
                      if(line.contains("Default"))
                          continue;
                      else{
                          String tmpT[] = line.split(":");
                          tmpT[1]=tmpT[1].replaceAll("−", "-");
                          auprc = Double.parseDouble(tmpT[1].trim());
                          perfLine = 0;
                      } 
                  }
                 
              }
              read.close();
      }
      catch(Exception e){
          e.printStackTrace();
      }
            
            if(!resultsWeka.containsKey("RF600PCT"))
                resultsWeka.put("RF600PCT", new ArrayList<>());
              resultsWeka.get("RF600PCT").add(auprc);
              resultsWeka.get("RF600PCT").add(auc);   
     
      //call weka models for FC features
      Classifier clsf = new J48();
      dataTrainValidation.deleteAttributeAt(0);
      dataTest.deleteAttributeAt(0);
     try{
            clsf.buildClassifier(dataTrainValidation);
            Evaluation eval = new Evaluation(dataTrainValidation);
            eval.evaluateModel(clsf, dataTest); 
             auprc = 0.0; auc = 0.0;
            for(int i=0;i<dataTrainValidation.classAttribute().numValues();i++){
                System.out.println("AUPRC"+i+": "+eval.areaUnderPRC(i));
                System.out.println("AUC"+i+": "+eval.areaUnderROC(i));
                auprc+=eval.areaUnderPRC(i); auc+=eval.areaUnderROC(i);
            }
            auprc/=dataTrainValidation.classAttribute().numValues();
            auc/=dataTrainValidation.classAttribute().numValues();
            /*System.out.println("AUPRC0: "+eval.areaUnderPRC(0));
            System.out.println("AUC0: "+eval.areaUnderROC(0));
            System.out.println("AUPRC1: "+eval.areaUnderPRC(1));
            System.out.println("AUC1: "+eval.areaUnderROC(1));*/
            System.out.println("Av.AUPRC J48: "+auprc);
            System.out.println("Av.AUC J48: "+auc);
            System.out.println(eval.toSummaryString("\nResults FC\n=====\n",false));
            
            if(!resultsWeka.containsKey("J48")){
                resultsWeka.put("J48", new ArrayList<>());
            }
            
            resultsWeka.get("J48").add(auprc);
             resultsWeka.get("J48").add(auc);
            
            clsf = new NaiveBayes();
            
             clsf.buildClassifier(dataTrainValidation);
            eval = new Evaluation(dataTrainValidation);
            eval.evaluateModel(clsf, dataTest);
             auprc = 0; auc = 0;
            for(int i=0;i<dataTrainValidation.classAttribute().numValues();i++){
                System.out.println("AUPRC"+i+": "+eval.areaUnderPRC(i));
                System.out.println("AUC"+i+": "+eval.areaUnderROC(i));
                auprc+=eval.areaUnderPRC(i); auc+=eval.areaUnderROC(i);
            }
            auprc/=dataTrainValidation.classAttribute().numValues();
            auc/=dataTrainValidation.classAttribute().numValues();
            System.out.println("Av.AUPRC NB: "+auprc);
            System.out.println("Av.AUC NB: "+auc);
            System.out.println(eval.toSummaryString("\nResults FC\n=====\n",false));
            
             if(!resultsWeka.containsKey("NB")){
                resultsWeka.put("NB", new ArrayList<>());
            }
            
            resultsWeka.get("NB").add(auprc);
             resultsWeka.get("NB").add(auc);
            
            Logistic clsf2 = new Logistic();
            String options2[] = weka.core.Utils.splitOptions("-M 10000");
            clsf2.setOptions(options2);
             clsf2.buildClassifier(dataTrainValidation);
            eval = new Evaluation(dataTrainValidation);
            eval.evaluateModel(clsf2, dataTest);
             auprc = 0; auc = 0;
            for(int i=0;i<dataTrainValidation.classAttribute().numValues();i++){
                System.out.println("AUPRC"+i+": "+eval.areaUnderPRC(i));
                System.out.println("AUC"+i+": "+eval.areaUnderROC(i));
                auprc+=eval.areaUnderPRC(i); auc+=eval.areaUnderROC(i);
            }
            auprc/=dataTrainValidation.classAttribute().numValues();
            auc/=dataTrainValidation.classAttribute().numValues();
            System.out.println("Av.AUPRC Log: "+auprc);
            System.out.println("Av.AUC Log: "+auc);
            System.out.println(eval.toSummaryString("\nResults FC\n=====\n",false));
            
            
             if(!resultsWeka.containsKey("Log")){
                resultsWeka.put("Log", new ArrayList<>());
            }
            
            resultsWeka.get("Log").add(auprc);
             resultsWeka.get("Log").add(auc);
            
            // String options1[] = weka.core.Utils.splitOptions("-G"); 
            MultilayerPerceptron clsf1 = new MultilayerPerceptron();
           //clsf1.setOptions(options1);
            
             clsf1.buildClassifier(dataTrainValidation);
            eval = new Evaluation(dataTrainValidation);
            eval.evaluateModel(clsf1, dataTest);
             auprc = 0; auc = 0;
            for(int i=0;i<dataTrainValidation.classAttribute().numValues();i++){
                System.out.println("AUPRC"+i+": "+eval.areaUnderPRC(i));
                System.out.println("AUC"+i+": "+eval.areaUnderROC(i));
                auprc+=eval.areaUnderPRC(i); auc+=eval.areaUnderROC(i);
            }
            auprc/=dataTrainValidation.classAttribute().numValues();
            auc/=dataTrainValidation.classAttribute().numValues();
            System.out.println("Av.AUPRC MLP: "+auprc);
            System.out.println("Av.AUC MLP: "+auc);
            System.out.println(eval.toSummaryString("\nResults FC\n=====\n",false));
            
             if(!resultsWeka.containsKey("MLP")){
                resultsWeka.put("MLP", new ArrayList<>());
            }
            
            resultsWeka.get("MLP").add(auprc);
             resultsWeka.get("MLP").add(auc);
            
            clsf = new KStar(); 
            
             clsf.buildClassifier(dataTrainValidation);
            eval = new Evaluation(dataTrainValidation);
            eval.evaluateModel(clsf, dataTest);
             auprc = 0; auc = 0;
            for(int i=0;i<dataTrainValidation.classAttribute().numValues();i++){
                System.out.println("AUPRC"+i+": "+eval.areaUnderPRC(i));
                System.out.println("AUC"+i+": "+eval.areaUnderROC(i));
                auprc+=eval.areaUnderPRC(i); auc+=eval.areaUnderROC(i);
            }
            auprc/=dataTrainValidation.classAttribute().numValues();
            auc/=dataTrainValidation.classAttribute().numValues();
            System.out.println("Av.AUPRC KStar: "+auprc);
            System.out.println("Av.AUC KStar: "+auc);
            System.out.println(eval.toSummaryString("\nResults FC\n=====\n",false));
            
            
             if(!resultsWeka.containsKey("KS")){
                resultsWeka.put("KS", new ArrayList<>());
            }
            
            resultsWeka.get("KS").add(auprc);
             resultsWeka.get("KS").add(auc);
            
            clsf = new DecisionStump();
            
             clsf.buildClassifier(dataTrainValidation);
            eval = new Evaluation(dataTrainValidation);
            eval.evaluateModel(clsf, dataTest);
             auprc = 0; auc = 0;
            for(int i=0;i<dataTrainValidation.classAttribute().numValues();i++){
                System.out.println("AUPRC"+i+": "+eval.areaUnderPRC(i));
                System.out.println("AUC"+i+": "+eval.areaUnderROC(i));
                auprc+=eval.areaUnderPRC(i); auc+=eval.areaUnderROC(i);
            }
            auprc/=dataTrainValidation.classAttribute().numValues();
            auc/=dataTrainValidation.classAttribute().numValues();
            System.out.println("Av.AUPRC DStump: "+auprc);
            System.out.println("Av.AUC DStump: "+auc);
            System.out.println(eval.toSummaryString("\nResults FC\n=====\n",false));
            
             if(!resultsWeka.containsKey("DSt")){
                resultsWeka.put("DSt", new ArrayList<>());
            }
            
            resultsWeka.get("DSt").add(auprc);
             resultsWeka.get("DSt").add(auc);
            
            clsf = new LMT();
            
             clsf.buildClassifier(dataTrainValidation);
            eval = new Evaluation(dataTrainValidation);
            eval.evaluateModel(clsf, dataTest);
             auprc = 0; auc = 0;
            for(int i=0;i<dataTrainValidation.classAttribute().numValues();i++){
                System.out.println("AUPRC"+i+": "+eval.areaUnderPRC(i));
                System.out.println("AUC"+i+": "+eval.areaUnderROC(i));
                auprc+=eval.areaUnderPRC(i); auc+=eval.areaUnderROC(i);
            }
            auprc/=dataTrainValidation.classAttribute().numValues();
            auc/=dataTrainValidation.classAttribute().numValues();
            System.out.println("Av.AUPRC LMT: "+auprc);
            System.out.println("Av.AUC LMT: "+auc);
            System.out.println(eval.toSummaryString("\nResults FC\n=====\n",false));
            
             if(!resultsWeka.containsKey("LMT")){
                resultsWeka.put("LMT", new ArrayList<>());
            }
            
            resultsWeka.get("LMT").add(auprc);
             resultsWeka.get("LMT").add(auc);
    }
     catch(Exception e){
         e.printStackTrace();
     }
    }
    
    
    
    public static void main(String args[]){
        
        long startTime = System.currentTimeMillis();
        HashMap<String,Integer> targets = new HashMap<>();
        HashMap<Integer,Integer> targetsToIndex = new HashMap<>();
        ApplicationSettings appset=new ApplicationSettings();
        appset.readSettings(new File(args[0]));
        appset.readPreference();
        Mappings fid=new Mappings();
        Mappings fidFull=new Mappings();
        Mappings fidTest=new Mappings();
        
        DataSetCreator datJ=new DataSetCreator(appset.viewInputPaths, appset.outFolderPath,appset);

        DataSetCreator datJFull=null;
        DataSetCreator datJFullComputation=null;
        DataSetCreator datJValid = null;
        DataSetCreator datJTest=null;
        DataSetCreator datJTrainValid=null;
        
        WekaDatasetLoader wdl = new WekaDatasetLoader();
        Instances data=null;
        Instances dataTrain=null;
        Instances dataValidation=null;
        Instances dataTrainValidation = null;
        Instances dataTest=null;
        
        InstancesFilter insfilt = null;
        RuleSet set = new RuleSet();
        
        data = NewFeatureConstructionPipline.loadMappingAndTargetIndex(appset,  fid,  fidFull,  fidTest , wdl, targets, targetsToIndex);
        HashMap<ArrayList<Instances>,ArrayList<DataSetCreator>> m = null;
        ArrayList<Instances> ret = null; 
        ArrayList<DataSetCreator> ret1 = null;
        m = NewFeatureConstructionPipline.createTrainTest(appset, fid, data, insfilt, datJ, datJFull, datJTest, datJValid,datJTrainValid);
        Iterator<ArrayList<Instances>> it = m.keySet().iterator();
        ret = it.next();
        ret1 = m.get(ret);
        dataTrain = ret.get(0); dataValidation = ret.get(1); dataTrainValidation = ret.get(2); dataTest = ret.get(3); 
        datJFull = ret1.get(0); datJ = ret1.get(1); datJTest = ret1.get(2); datJValid = ret1.get(3); datJTrainValid = ret1.get(4);
        datJFullComputation = ret1.get(5);
        RedescriptionSet resultSetOut = new RedescriptionSet();
        RuleReader rrResultOut = new RuleReader();
        NewFeatureConstructionPipline.createDescriptiveRulesAndRedescriptions(appset, fid, fidTest, datJTest, datJFullComputation , targets, targetsToIndex , resultSetOut, rrResultOut);

       ArrayList<Rule> rulesSupervised = new ArrayList<>();
       NewFeatureConstructionPipline.createSupervisedRules(appset, fid, datJTest, wdl, rulesSupervised);
       
       ArrayList<Rule> subgroups = new ArrayList<>();
       String trainValidationPath = "";
       if(appset.system.equals("windows"))
            trainValidationPath = appset.outFolderPath+"\\TrainValidation.arff";
       else trainValidationPath =appset.outFolderPath+"/TrainValidation.arff";
       String testPath = "";
       
       if(appset.system.equals("windows"))
            testPath = appset.outFolderPath+"\\Test.arff";
       else testPath =appset.outFolderPath+"/Test.arff";
 
         Cleaner c = new Cleaner();
       c.clean(trainValidationPath);
       c.clean(testPath);
       
       NewFeatureConstructionPipline.createSubgroups(appset, fid, datJTest, wdl, subgroups, trainValidationPath, testPath);
       System.out.println("MAIN: "+rulesSupervised.size()+" supervised rules, "+resultSetOut.redescriptions.size()+" redescriptions, "+rrResultOut.rules.size()+" descriptive rules,"+subgroups.size()+" subgroups");
       
        ArrayList<Instances> ret2 = NewFeatureConstructionPipline.createFilesForPredictivityTesting(appset, fidFull, fid, wdl, rrResultOut, resultSetOut, rulesSupervised, subgroups, data, datJ, datJFull, trainValidationPath, testPath);
       
        //String inputR = "C:/Users/Matej/OneDrive/Dokumenti/compFeatures2.R";
        String inputR = appset.outFolderPath+"/compFeatures2.R";
        //String datasetIn = "Abalone";
        //String datasetIn = "Arrythmia";
        String datasetIn = args[1].trim();
        String rOutput = appset.outFolderPath.replaceAll("\\\\", "/");
        rOutput+="/";
        //String libraryPath = "C:/Users/Matej/OneDrive/Dokumenti/R/win-library/4.1";
        String libraryPath = "/home/mmihelcic/R/x86_64-pc-linux-gnu-library/4.2";
        String trainValidationAllPath = appset.outFolderPath+"\\TrainValidationAllFeatures.arff";
      
        if(!appset.system.equals("windows"))
            trainValidationAllPath = appset.outFolderPath+"/TrainValidationAllFeatures.arff";
          c.clean(trainValidationAllPath);
        String testAllPath = appset.outFolderPath+"\\TestAllFeatures.arff";
        if(!appset.system.equals("windows"))
                testAllPath = appset.outFolderPath+"/TestAllFeatures.arff";
        c.clean(testAllPath);
        String trainValidationPathR = trainValidationAllPath.replaceAll("\\\\", "/");   
        String testPathR = testAllPath.replaceAll("\\\\", "/");
        System.out.println("TR: "+trainValidationPathR);
        System.out.println("TS: "+testPathR);
        System.out.println("RO: "+rOutput);
        NewFeatureConstructionPipline.runRFunctions(inputR, trainValidationPathR, testPathR, datasetIn, rOutput, libraryPath);
        
        HashMap<String,ArrayList<Double>> resultsWeka = new HashMap<>();
        String trainClass = "";
         String testClass = "";
        //original
        
         if(appset.system.equals("windows"))
            trainClass = appset.outFolderPath+"\\TrainValidation.arff";
       else trainClass =appset.outFolderPath+"/TrainValidation.arff";
      
       
       if(appset.system.equals("windows"))
            testClass = appset.outFolderPath+"\\Test.arff";
       else testClass =appset.outFolderPath+"/Test.arff";
       
       System.out.println(trainClass);
       System.out.println(testClass);
        
        NewFeatureConstructionPipline.trainTestClassifiers(appset, wdl, trainClass, testClass, resultsWeka);
        
        //original boruta + svetnik original
        
         if(appset.system.equals("windows"))
            trainClass = appset.outFolderPath+"\\OriginalBNR"+datasetIn+".arff";
       else trainClass =appset.outFolderPath+"/OriginalBNR"+datasetIn+".arff";
      
       
       if(appset.system.equals("windows"))
            testClass = appset.outFolderPath+"\\OriginalBNRTest"+datasetIn+".arff";
       else testClass =appset.outFolderPath+"/OriginalBNRTest"+datasetIn+".arff";
       
       System.out.println(trainClass);
       System.out.println(testClass);
        
        NewFeatureConstructionPipline.trainTestClassifiers(appset, wdl, trainClass, testClass, resultsWeka);

        //all features boruta + svetnik
        if(appset.system.equals("windows"))
            trainClass = appset.outFolderPath+"\\OrigAllBNR"+datasetIn+".arff";
       else trainClass =appset.outFolderPath+"/OrigAllBNR"+datasetIn+".arff";
      
       
       if(appset.system.equals("windows"))
            testClass = appset.outFolderPath+"\\OrigAllBNRTest"+datasetIn+".arff";
       else testClass =appset.outFolderPath+"/OrigAllBNRTest"+datasetIn+".arff";
        
       System.out.println(trainClass);
       System.out.println(testClass);
       
        NewFeatureConstructionPipline.trainTestClassifiers(appset, wdl, trainClass, testClass, resultsWeka);
        
        //orig + S rules features boruta + svetnik
        if(appset.system.equals("windows"))
            trainClass = appset.outFolderPath+"\\OrigSRulesBNR"+datasetIn+".arff";
       else trainClass =appset.outFolderPath+"/OrigSRulesBNR"+datasetIn+".arff";
      
       
       if(appset.system.equals("windows"))
            testClass = appset.outFolderPath+"\\OrigSRulesBNRTest"+datasetIn+".arff";
       else testClass =appset.outFolderPath+"/OrigSRulesBNRTest"+datasetIn+".arff";
        
       System.out.println(trainClass);
       System.out.println(testClass);
       
        NewFeatureConstructionPipline.trainTestClassifiers(appset, wdl, trainClass, testClass, resultsWeka);
        
              //orig + D rules features boruta + svetnik
        if(appset.system.equals("windows"))
            trainClass = appset.outFolderPath+"\\OrigRulesBNR"+datasetIn+".arff";
       else trainClass =appset.outFolderPath+"/OrigRulesBNR"+datasetIn+".arff";
      
       
       if(appset.system.equals("windows"))
            testClass = appset.outFolderPath+"\\OrigRulesBNRTest"+datasetIn+".arff";
       else testClass =appset.outFolderPath+"/OrigRulesBNRTest"+datasetIn+".arff";
        
        System.out.println(trainClass);
        System.out.println(testClass);
       
        NewFeatureConstructionPipline.trainTestClassifiers(appset, wdl, trainClass, testClass, resultsWeka);
        
               //orig + Reds features boruta + svetnik
        if(appset.system.equals("windows"))
            trainClass = appset.outFolderPath+"\\OrigRedsBNR"+datasetIn+".arff";
       else trainClass =appset.outFolderPath+"/OrigRedsBNR"+datasetIn+".arff";
      
       
       if(appset.system.equals("windows"))
            testClass = appset.outFolderPath+"\\OrigRedsBNRTest"+datasetIn+".arff";
       else testClass =appset.outFolderPath+"/OrigRedsBNRTest"+datasetIn+".arff";
        
       System.out.println(trainClass);
       System.out.println(testClass);
       
        NewFeatureConstructionPipline.trainTestClassifiers(appset, wdl, trainClass, testClass, resultsWeka);
        
                 //orig + Subg features boruta + svetnik
        if(appset.system.equals("windows"))
            trainClass = appset.outFolderPath+"\\OrigSubgBNR"+datasetIn+".arff";
       else trainClass =appset.outFolderPath+"/OrigSubgBNR"+datasetIn+".arff";
      
       
       if(appset.system.equals("windows"))
            testClass = appset.outFolderPath+"\\OrigSubgBNRTest"+datasetIn+".arff";
       else testClass =appset.outFolderPath+"/OrigSubgBNRTest"+datasetIn+".arff";
        
       System.out.println(trainClass);
       System.out.println(testClass);
       
        NewFeatureConstructionPipline.trainTestClassifiers(appset, wdl, trainClass, testClass, resultsWeka);
        
        try{
            String output = "";
            
            if(appset.system.equals("windows"))
            output = appset.outFolderPath+"\\ClassifierScore"+datasetIn+".txt";
       else output =appset.outFolderPath+"/Classifier"+datasetIn+".txt";
            
             FileWriter fw = new FileWriter(output);
             
             fw.write("FeatOriginalAUPRC FeatOriginalSelAUPRC FeatAllSelAUPRC FeatOrigSRulesSelAUPRC FeatOrigDRulesSelAUPRC FeatOrigRedsSelAUPRC FeatOrigSubgSelAUPRC FeatOriginalAUC FeatOriginalSelAUC FeatAllSelAUC FeatOrigSRulesSelAUC FeatOrigDRulesSelAUC FeatOrigRedsSelAUC FeatOrigSubgSelAUC\n\n");
             
             Iterator<String> it1 = resultsWeka.keySet().iterator();
             
             while(it1.hasNext()){
                 String a = it1.next();
                 fw.write(a+" "+resultsWeka.get(a).get(0)+" "+resultsWeka.get(a).get(2)+" "+resultsWeka.get(a).get(4)+" "+resultsWeka.get(a).get(6)+" "+resultsWeka.get(a).get(8)+" "+resultsWeka.get(a).get(10)+" "+resultsWeka.get(a).get(12)+" "+resultsWeka.get(a).get(1)+" "+resultsWeka.get(a).get(3)+" "+resultsWeka.get(a).get(5)+" "+resultsWeka.get(a).get(7)+" "+resultsWeka.get(a).get(9)+" "+resultsWeka.get(a).get(11)+" "+resultsWeka.get(a).get(13)+"\n");
             }
             fw.close();
        }
        catch(Exception e){
            e.printStackTrace();
        }
        
        
    }
}
