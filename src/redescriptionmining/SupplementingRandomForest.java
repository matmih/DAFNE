/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package redescriptionmining;

import gnu.trove.iterator.TIntIterator;
import gnu.trove.set.hash.TIntHashSet;
import java.io.BufferedReader;
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
import java.util.TreeMap;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.KStar;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.LMT;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSink;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author matej
 */
public class SupplementingRandomForest {
    public static void main(String[] args) {

        //1. compute pareto optimal front of rules
        //2. visualizations
        //3. add several initial clustering methods (using target-descriptive and both), future work
        //4. add settings element for network approach, future work...
        //5. if return redescription = all return all redescriptions and do not compute rule score (perhaps do not compute vizaulization set also?)
        //6. multi-view approach

                long startTime = System.currentTimeMillis();
                
                HashMap<String,Integer> targets = new HashMap<>();
                HashMap<Integer,Integer> targetsToIndex = new HashMap<>();
                

        ApplicationSettings appset=new ApplicationSettings();
        appset.readSettings(new File(args[0]));
             appset.readPreference(); 
        System.out.println("Num targets: "+appset.numTargets);
        System.out.println("Num trees in RS: "+appset.numTreesinForest);
        System.out.println("Average tree depth in RS: "+appset.aTreeDepth);
        System.out.println("Allow left side rule negation: "+appset.leftNegation);
        System.out.println("Allow right side rule negation: "+appset.rightNegation);
        System.out.println("Allow left side rule disjunction: "+appset.leftDisjunction);
        System.out.println("Allow right side rule disjunction: "+appset.rightDisjunction);
        System.out.println("Types of LSTrees: "+appset.treeTypes.get(0));
        System.out.println("Types of RSTrees: "+appset.treeTypes.get(1));
        System.out.println("Use Network information: "+appset.useNC.toString());
        System.out.println("Spatial matrix: "+appset.spatialMatrix.toString());
        System.out.println("Spatial measure: "+appset.spatialMeasures.toString());
        
        
        System.out.println("Attribute importance gen: ");
        for(int i=0;i<appset.attributeImportanceGen.size();i++)
              System.out.print(appset.attributeImportanceGen.get(i)+" ");
        System.out.println();
        
        System.out.println("Important attributes: ");
        for(int i=0;i<appset.importantAttributes.size();i++){
            for(int j=0;j<appset.importantAttributes.get(i).size();j++){
                for(int k=0;k<appset.importantAttributes.get(i).get(j).size();k++){
                    if(k<appset.importantAttributes.get(i).get(j).size())
                        System.out.print(appset.importantAttributes.get(i).get(j).get(k)+" , ");
                }
                System.out.print(" + ");
            }
        System.out.println();
        }
        //appset.importantAttributes

        Mappings fid=new Mappings();
        Mappings fidFull=new Mappings();
        Mappings fidTest=new Mappings();
        
        DataSetCreator datJ=new DataSetCreator(appset.viewInputPaths, appset.outFolderPath,appset);

        DataSetCreator datJFull=null;
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
        
        //DataSetCreator datJ=new DataSetCreator(appset.viewInputPaths.get(0)/*appset.outFolderPath+"\\input1.arff"*/,appset.viewInputPaths.get(1)/*appset.outFolderPath+"\\input2.arff"*/ , appset.outFolderPath);
        //fid.createIndex(appset.outFolderPath+"\\Jinput.arff");
        
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
           
           if(appset.useSplitTesting==true){
               
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
                     DataSink.write(outputFilenamePred, tmp); //write the predictive test set
                    
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
                     DataSink.write(outputFilenamePred, tmp); //write the predictive test set
                    
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
                     
                     String outputFilename = "";
                    if(appset.system.equals("windows"))
                        outputFilename= appset.outFolderPath+"\\JinputTest.arff";
                    else outputFilename= appset.outFolderPath+"/JinputTest.arff";
                     DataSink.write(outputFilename, dataTest);
                     if(appset.system.equals("windows"))
                        outputFilename = appset.outFolderPath+"\\JinputTrainValidation.arff";
                     else  outputFilename = appset.outFolderPath+"/JinputTrainValidation.arff";
                     DataSink.write(outputFilename, dataTrainValidation);
                     
                     if(appset.system.equals("windows"))
                                outputFilename = appset.outFolderPath+"\\JinputValidation.arff";
                     else  outputFilename = appset.outFolderPath+"/JinputValidation.arff";
                     DataSink.write(outputFilename, dataValidation);
                     
                      if(appset.system.equals("windows"))
                             outputFilename = appset.outFolderPath+"\\JinputTrain.arff";
                      else  outputFilename = appset.outFolderPath+"/JinputTrain.arff";
                     DataSink.write(outputFilename, dataTrain);
                     
               }
               catch(Exception e){
                   e.printStackTrace();
               }
               
               
              ArrayList<DataSetCreator> rDat = new ArrayList<>();
               datJFull=datJ;
               datJTest = datJ;
               
                   if(appset.system.equals("windows")){
                      // datJFull = new DataSetCreator(appset.outFolderPath+"\\Jinput.arff");
                        datJ = new DataSetCreator(appset.outFolderPath+"\\JinputTrain.arff");
                        datJTest = new DataSetCreator(appset.outFolderPath+"\\JinputTest.arff");
                        datJValid = new DataSetCreator(appset.outFolderPath+"\\JinputValidation.arff");
                        datJTrainValid = new DataSetCreator(appset.outFolderPath+"\\JinputTrainValidation.arff");
                   }
                   else{
                         datJ = new DataSetCreator(appset.outFolderPath+"/JinputTrain.arff");
                        datJTest = new DataSetCreator(appset.outFolderPath+"/JinputTest.arff");
                        datJValid = new DataSetCreator(appset.outFolderPath+"/JinputValidation.arff");
                        datJTrainValid = new DataSetCreator(appset.outFolderPath+"/JinputTrainValidation.arff");
                   }
                   
                    try{
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
                         datJ.W2indexs.addAll(datJFull.W2indexs);//ovo nije OK
                         datJTest.W2indexs.addAll(datJFull.W2indexs);
                         datJValid.W2indexs.addAll(datJFull.W2indexs);   
                         datJTrainValid.W2indexs.addAll(datJFull.W2indexs);
            
         fid.clearMaps();
         if(appset.system.equals("windows"))
             fid.createIndex(appset.outFolderPath+"\\JinputTrain.arff");
         else fid.createIndex(appset.outFolderPath+"/JinputTrain.arff");
          System.out.println("Full: "+datJFull.numExamples);
        System.out.println("Part: "+datJ.numExamples);
        System.out.println("Mapping: "+fid.attId.size());
        }
        
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
            
                   
         elemFreq=new int[datJ.numExamples];
         attrFreq=new int[datJ.schema.getNbAttributes()];  
            
          System.out.println("Number of redescriptions: "+appset.numInitial);
        
        redScores=new ArrayList<>(appset.numInitial);
        redScoresAtt=new ArrayList<>(appset.numInitial);
        redDistCoverage=new ArrayList<>(appset.numInitial);
        redDistCoverageAt=new ArrayList<>(appset.numInitial);
        if(appset.useNetworkAsBackground==true)
              redDistNetwork=new ArrayList<>(appset.numInitial);
         targetAtScore=null;
        //double Statistics[]={0.0,0.0,0.0};//previousMedian - 0, numberIterationsStable - 1, minDifference - 2
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
            

            
        /*if(appset.attributeImportance!=0)
            targetAtScore = new ArrayList<>(appset.numRetRed);
        
        for(int z=0;z<appset.numRetRed;z++){
            redScores.add(Double.NaN);
            redScoresAtt.add(Double.NaN);
        }*/
      }
                    
        NHMCDistanceMatrix nclMatInit=null;
        if(appset.distanceFilePaths.size()>0){
            nclMatInit=new NHMCDistanceMatrix(datJ.numExamples,appset);
            nclMatInit.loadDistance(new File(appset.distanceFilePaths.get(0)), fid);
            if(appset.distanceFilePaths.size()>0){
             nclMatInit.resetFile(new File(appset.outFolderPath+"\\distances.csv"));
             nclMatInit.writeToFile(new File(appset.outFolderPath+"\\distances.csv"), fid,appset);
            }
            else{
                nclMatInit.resetFile(new File(appset.outFolderPath+"/distances.csv"));
                nclMatInit.writeToFile(new File(appset.outFolderPath+"/distances.csv"), fid,appset);
            }
             nclMatInit=null;
        }
        
        for(int runTest=0;runTest<appset.numRandomRestarts;runTest++){  
            
           System.out.println("Random restart no: "+(runTest+1)); 
            
          DataSetCreator datJInit=null;
          
       if(!appset.useSplitTesting){ 
        if(appset.initClusteringFileName.equals("")){
            if(appset.system.equals("windows")){
                datJInit = new DataSetCreator(appset.outFolderPath+"\\Jinput.arff");
            }
            else{
                datJInit = new DataSetCreator(appset.outFolderPath+"/Jinput.arff");
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
                datJInit = new DataSetCreator(appset.outFolderPath+"\\JinputTrain.arff");
            else
                datJInit = new DataSetCreator(appset.outFolderPath+"/JinputTrain.arff");
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
        
        datJInit.W2indexs.addAll(datJ.W2indexs);
        
        if(appset.initClusteringFileName.equals(""))
            datJInit.initialClusteringGen1(appset.outFolderPath,appset,datJ.schema.getNbDescriptiveAttributes(),r);
        
        SettingsReader initSettings=new SettingsReader();
        
        if(appset.initClusteringFileName.equals(""))
             if(appset.system.equals("windows"))
                 initSettings.setDataFilePath(appset.outFolderPath+"\\JinputInitial.arff");
             else
                  initSettings.setDataFilePath(appset.outFolderPath+"/JinputInitial.arff");
        else{
            if(appset.system.equals("windows"))
                 initSettings.setDataFilePath(appset.outFolderPath+"\\"+appset.initClusteringFileName);
            else
                initSettings.setDataFilePath(/*appset.outFolderPath+"/"+*/appset.initClusteringFileName);
        }
        
        //fid.printMapping();
        //make initial cluster analysis
        
        System.out.println("WIndexes size: "+datJ.W2indexs.size());
        
       /* DataSetCreator datJInit=new DataSetCreator(appset.outFolderPath+"\\Jinput.arff");
                try{
        datJInit.readDataset();
        }
        catch(IOException e){
            e.printStackTrace();
        }*/
        
       // datJInit.W2indexs.addAll(datJ.W2indexs);
       // datJInit.initialClusteringGen1(appset.outFolderPath,appset, datJ.schema.getNbDescriptiveAttributes(),r);
        //datJInit.initialClustering(appset.outFolderPath,appset);
        //datJInit.initialClusteringCategorical(appset.outFolderPath);
        
       /* SettingsReader initSettings=new SettingsReader();
        initSettings.setDataFilePath(appset.outFolderPath+"\\JinputInitial.arff");*/
        if(appset.system.equals("windows"))
             initSettings.setPath(appset.outFolderPath+"\\view1.s");
        else
             initSettings.setPath(appset.outFolderPath+"/view1.s");
        //initSettings.createInitialSettings1(1, datJ.W2indexs.get(0), datJInit.schema.getNbAttributes(), appset);
        System.out.println("distance file size: "+appset.distanceFilePaths.size()+"");
        System.out.println("use nc: "+appset.useNC.size());
        if(appset.useNC.get(0) == false)
             initSettings.createInitialSettingsGen(0, 3, datJ.W2indexs.get(0), datJ.schema.getNbAttributes(), appset,1);
        else
             initSettings.createInitialSettingsGen(0, 4, datJ.W2indexs.get(0), datJ.schema.getNbAttributes(), appset,1);
            
        ClusProcessExecutor exec=new ClusProcessExecutor();

        //RunInitW1S1
        exec.run(appset.javaPath,appset.clusPath ,appset.outFolderPath,"view1.s",0, appset.clusteringMemory);//was 1 before for rules
        System.out.println("Process 1 side 1 finished!");
         
        //read the rules obtained from first attribute set
         // String input1="C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\unctad.out";
          String input1="";
          if(appset.system.equals("windows"))
             input1=appset.outFolderPath+"\\view1.out";
          else
              input1=appset.outFolderPath+"/view1.out"; 
           
           rr1.extractRules(input1,fid,datJInit,appset);
           
         //reading arff file
         
           if(appset.distanceFilePaths.size()>1){
            nclMatInit=new NHMCDistanceMatrix(datJ.numExamples,appset);
            nclMatInit.loadDistance(new File(appset.distanceFilePaths.get(1)), fid);
            if(appset.system.equals("windows")){
             nclMatInit.resetFile(new File(appset.outFolderPath+"\\distances.csv"));
             nclMatInit.writeToFile(new File(appset.outFolderPath+"\\distances.csv"), fid,appset);
            }
            else{
                nclMatInit.resetFile(new File(appset.outFolderPath+"/distances.csv"));
                nclMatInit.writeToFile(new File(appset.outFolderPath+"/distances.csv"), fid,appset);
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
        if(datJ.W2indexs.size()>1)
            initSettings.createInitialSettingsGen(1, datJ.W2indexs.get(0)+1, datJ.W2indexs.get(1), datJ.schema.getNbAttributes(), appset,1);
        else
            initSettings.createInitialSettingsGen(1, datJ.W2indexs.get(0)+1, datJInit.schema.getNbAttributes(), datJ.schema.getNbAttributes(), appset,1);
        
        exec.run(appset.javaPath,appset.clusPath, appset.outFolderPath, "view2.s", 0,appset.clusteringMemory);//was 1 before
        System.out.println("Process 1 side 2 finished!");

        //read the rules obtained from first attribute set
       if(appset.system.equals("windows"))
        input1=appset.outFolderPath+"\\view2.out";
       else
           input1=appset.outFolderPath+"/view2.out";
        rr.extractRules(input1,fid,datJInit,appset);
           
           FileDeleter delTmp=new FileDeleter();
           if(appset.system.equals("windows"))
                delTmp.setPath(appset.outFolderPath+"\\JinputInitial.arff");
           else
               delTmp.setPath(appset.outFolderPath+"/JinputInitial.arff");
           delTmp.delete();
        
          if(appset.system.equals("windows"))
           initSettings.setPath(appset.outFolderPath+"\\view1.s");
          else
              initSettings.setPath(appset.outFolderPath+"/view1.s");
          // initSettings.createInitialSettings(1, datJ.W2indexs.get(0), datJ.schema.getNbAttributes(), appset);
           //initSettings.createInitialSettingsGen(0, 3, datJ.W2indexs.get(0), datJ.schema.getNbAttributes(), appset);
          if(appset.system.equals("windows")) 
             initSettings.setPath(appset.outFolderPath+"\\view2.s");
          else
             initSettings.setPath(appset.outFolderPath+"/view2.s");
          // initSettings.createInitialSettings(2, datJ.W2indexs.get(0), datJ.schema.getNbAttributes(), appset);
          // initSettings.createInitialSettingsGen(0, datJ.W2indexs.get(0)+1, datJ.W2indexs.get(1), datJ.schema.getNbAttributes(), appset);
         
           datJInit=null;        
           
        int leftSide=1, rightSide=0;//set left to 1 when computing lf, otherwise right
        int leftSide1=0, rightSide1=1; //left, right side for Side 2
        int it=0;
        Jacard js=new Jacard();
        Jacard jsN[]=new Jacard[3];
        
        for(int i=0;i<jsN.length;i++)
            jsN[i]=new Jacard();
            
        //ArrayList<Redescription> redescriptions=new ArrayList<Redescription>();
       
        int newRedescriptions=1;
        int numIter=0;
        int RunInd=0;
       
        int naex=datJ.numExamples;
        
        //add arrayList of view rules
        ArrayList<RuleReader> readers=new ArrayList<>();
        int oldRIndex[]={0};
        
        NHMCDistanceMatrix nclMat=null;
        if((appset.distanceFilePaths.size()>0 || appset.useNC.get(0)==true) && appset.networkInit==false)
            nclMat=new NHMCDistanceMatrix(datJ.numExamples,appset);
        NHMCDistanceMatrix nclMat1=null;
        if((appset.distanceFilePaths.size()>1 || appset.useNC.get(1)==true) && appset.networkInit==false)
            nclMat1=new NHMCDistanceMatrix(datJ.numExamples,appset);
       
        if(appset.useNetworkAsBackground==true)
            appset.networkInit=false;
        
        if(appset.useNC.size()>=2 && appset.useNC.get(1) == true){
            if(appset.system.equals("windows")) 
                initSettings.setPath(appset.outFolderPath+"\\view2.s");
            else
                initSettings.setPath(appset.outFolderPath+"/view2.s");
         if(appset.useNC.size()>2)
            initSettings.createInitialSettingsGenN(1, datJ.W2indexs.get(0)+1, datJ.W2indexs.get(1), datJ.schema.getNbAttributes(), appset);
         else
            initSettings.createInitialSettingsGenN(1, datJ.W2indexs.get(0)+1, datJ.schema.getNbAttributes()+1, datJ.schema.getNbAttributes(), appset); 
        }
        if(appset.useNC.size()>1 && appset.useNC.get(0) == true){
             if(appset.system.equals("windows")) 
                initSettings.setPath(appset.outFolderPath+"\\view1.s");
             else
                initSettings.setPath(appset.outFolderPath+"/view1.s");
         initSettings.createInitialSettingsGenN(1, 4, datJ.W2indexs.get(0), datJ.schema.getNbAttributes(), appset);
        }
        
        /*if(appset.networkInit==true){
            appset.useNC.set(0, false);
            datJ=new DataSetCreator(appset.viewInputPaths, appset.outFolderPath,appset);
            appset.useNC.set(0, true);
        }*/
        
        while(newRedescriptions!=0 && RunInd<appset.numIterations){
            
       DataSetCreator dsc=null;//new DataSetCreator(appset.outFolderPath+"\\Jinput.arff");
       DataSetCreator dsc1=null;//new DataSetCreator(appset.outFolderPath+"\\Jinput.arff");
       
       rr.setSize();
       rr1.setSize();
       
       //read dataset for cicle 1
        /*try{
        dsc.readDataset();
        }
        catch(IOException e){
            e.printStackTrace();
        }

        //read dataset for cicle 2
        try{
        dsc1.readDataset();
        }
        catch(IOException e){
            e.printStackTrace();
        }*/

       //dsc.data.getNbRows();
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
                             nclMat.loadDistance(new File(appset.distanceFilePaths.get(1)), fid);
                             //nclMat.writeToFile(new File(appset.outFolderPath+"\\distance.csv"), fid);
                    }
                     else if(appset.computeDMfromRules==true){
                             nclMat.computeDistanceMatrix(rr1, fid, appset.maxDistance, datJ.numExamples);
                     }
                   }
                }
                 if(appset.useNC.get(0)==true && appset.networkInit==false && appset.useNetworkAsBackground==false){
                 if(rightSide==1){
                    if(appset.distanceFilePaths.size()>=2){
                             nclMat.loadDistance(new File(appset.distanceFilePaths.get(0)), fid);
                            // nclMat.writeToFile(new File(appset.outFolderPath+"\\distance.csv"), fid);
                    }
                     else if(appset.computeDMfromRules==true){
                             nclMat.computeDistanceMatrix(rr, fid, appset.maxDistance, datJ.numExamples);
                     }
                   }
                 }
                  if(appset.useNC.get(1)==true && appset.networkInit==false && appset.useNetworkAsBackground==false){
                if(leftSide1==1){
                    if(appset.distanceFilePaths.size()>=1){
                             nclMat1.loadDistance(new File(appset.distanceFilePaths.get(1)), fid);
                            // nclMat1.writeToFile(new File(appset.outFolderPath+"\\distance.csv"), fid);
                    }
                     else if(appset.computeDMfromRules==true){
                             nclMat1.computeDistanceMatrix(rr1, fid, appset.maxDistance, datJ.numExamples);
                     }
                }
                  }
                   if(appset.useNC.get(0)==true && appset.networkInit==false && appset.useNetworkAsBackground==false){
                    if(rightSide1==1){
                    if(appset.distanceFilePaths.size()>=2){
                             nclMat1.loadDistance(new File(appset.distanceFilePaths.get(0)), fid);
                            // nclMat1.writeToFile(new File(appset.outFolderPath+"\\distance.csv"), fid);
                    }
                     else if(appset.computeDMfromRules==true){
                             nclMat1.computeDistanceMatrix(rr, fid, appset.maxDistance, datJ.numExamples);
                     }
                  }
                }
            }
            
            /*System.out.println("Computing cuttof");
            cuttof1=rr1.findCutoff(naex, startPerc, endPerc, oldIndexRR1, endIndexRR1, minCovElements1,maxCovElements1, appset.minSupport,appset.maxSupport,appset.numTargets);
            cuttof=rr.findCutoff(naex, startPerc, endPerc, oldIndexRR, endIndexRR, minCovElements,maxCovElements, appset.minSupport,appset.maxSupport,appset.numTargets);
   */
            
            
           // System.out.println("Cuttof: "+minCovElements[0]);
           // System.out.println("Cuttof1: "+minCovElements1[0]);

            /*if(cuttof==-1 || cuttof1==-1){
                //System.out.println("Rule count: 0");
               // return;
                continue;
            }*/
           
          
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
                             dsc=new DataSetCreator(appset.outFolderPath+"\\JinputTrain.arff");
                             dsc1=new DataSetCreator(appset.outFolderPath+"\\JinputTrain.arff");
                         }
                     else{
                             dsc=new DataSetCreator(appset.outFolderPath+"/JinputTrain.arff");
                             dsc1=new DataSetCreator(appset.outFolderPath+"/JinputTrain.arff"); 
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
             /*if(z!=0)
                 set.changeSeed();*/
             
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
       //modify settings
            /* for(int i=oldIndexRR1;i<endIndexRR1;i++) //do on the fly when reading rules
                    if(rr1.rules.get(i).elements.size()<=naex*endPerc && rr1.rules.get(i).elements.size()>=naex*startPerc && rr1.rules.get(i).elements.size()>=minCovElements1[0] && rr1.rules.get(i).elements.size()<=maxCovElements1[0]) //do parameters analysis in this step
                        nARules++;
             set.ModifySettings(nARules,dsc.schema.getNbAttributes());*/
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
                 /* if(z!=0)
                     set.changeSeed();*/

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
                 
                 /*for(int i=oldIndexRR;i<endIndexRR;i++) //do on the fly when reading rules
                        if(rr.rules.get(i).elements.size()<=naex*endPerc && rr.rules.get(i).elements.size()>=naex*startPerc && rr.rules.get(i).elements.size()>=minCovElements[0] && rr.rules.get(i).elements.size()<=maxCovElements[0])
                            nARules++;
                set.ModifySettings(nARules,dsc1.schema.getNbAttributes());*/
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
             /* if(z!=0)
                 set1.changeSeed();*/

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
             
            /* for(int i=oldIndexRR1;i<endIndexRR1;i++)
                  if(rr1.rules.get(i).elements.size()<=naex*endPerc && rr1.rules.get(i).elements.size()>=naex*startPerc && rr1.rules.get(i).elements.size()>=minCovElements1[0] && rr1.rules.get(i).elements.size()<=maxCovElements1[0])
                       nARules1++;
             set1.ModifySettings(nARules1,dsc.schema.getNbAttributes());*/
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
              /*if(z!=0)
                    set1.changeSeed();*/

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
              
              /*for(int i=oldIndexRR;i<endIndexRR;i++)
                  if(rr.rules.get(i).elements.size()<=naex*endPerc && rr.rules.get(i).elements.size()>=naex*startPerc && rr.rules.get(i).elements.size()>=minCovElements[0] && rr.rules.get(i).elements.size()<=maxCovElements[0])
                        nARules1++;
              set1.ModifySettings(nARules1,dsc1.schema.getNbAttributes());*/
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
                 dsc.modifyDatasetS(startIndexRR1,endTmp,rr1,appset.outFolderPath+"\\Jinputnew.arff",fid,appset);
             }
             else{
                dsc.modifyDatasetS(startIndexRR1,endTmp,rr1,appset.outFolderPath+"/Jinputnew.arff",fid,appset); 
             }
                //dsc.modifyDatasetS(nARules, startPerc, endPerc, oldIndexRR1, endIndexRR1, minCovElements1[0],maxCovElements1[0], rr1,appset.outFolderPath+"\\Jinputnew.arff"/*"C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\UNCTADall3testnew.arff"*/,fid);
         else if(appset.treeTypes.get(1)==0/*appset.typeOfRSTrees==0*/)
             if(appset.system.equals("windows")){ 
                 dsc.modifyDatasetCat(startIndexRR1,endTmp,rr1,appset.outFolderPath+"\\Jinputnew.arff",fid,appset);
             }
             else{
                dsc.modifyDatasetCat(startIndexRR1,endTmp,rr1,appset.outFolderPath+"/Jinputnew.arff",fid,appset); 
             }
                 //dsc.modifyDatasetCat(nARules, startPerc, endPerc, oldIndexRR1, endIndexRR1, minCovElements1[0],maxCovElements1[0], rr1,appset.outFolderPath+"\\Jinputnew.arff"/*"C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\UNCTADall3testnew.arff"*/,fid);
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
                    dsc.modifyDatasetS(startIndexRR,endTmp,rr,appset.outFolderPath+"\\Jinputnew.arff",fid,appset);
                 }
                 else{
                     dsc.modifyDatasetS(startIndexRR,endTmp,rr,appset.outFolderPath+"/Jinputnew.arff",fid,appset); 
                 }
                //dsc.modifyDatasetS(nARules, startPerc, endPerc, oldIndexRR, endIndexRR, minCovElements[0],maxCovElements[0], rr,appset.outFolderPath+"\\Jinputnew.arff"/*"C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\UNCTADall3testnew.arff"*/,fid);
             else if(appset.treeTypes.get(0)==0/*appset.typeOfLSTrees==0*/)
                  if(appset.system.equals("windows")){ 
                    dsc.modifyDatasetCat(startIndexRR,endTmp ,rr,appset.outFolderPath+"\\Jinputnew.arff",fid,appset);
                  }
                  else{
                     dsc.modifyDatasetCat(startIndexRR,endTmp ,rr,appset.outFolderPath+"/Jinputnew.arff",fid,appset); 
                  }
               // dsc.modifyDatasetCat(nARules, startPerc, endPerc, oldIndexRR, endIndexRR, minCovElements[0],maxCovElements[0], rr,appset.outFolderPath+"\\Jinputnew.arff"/*"C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\UNCTADall3testnew.arff"*/,fid);
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
                    dsc1.modifyDatasetS(startIndexRR1,endTmp, rr1,appset.outFolderPath+"\\Jinputnew1.arff",fid,appset);
                }
                else{
                   dsc1.modifyDatasetS(startIndexRR1,endTmp, rr1,appset.outFolderPath+"/Jinputnew1.arff",fid,appset); 
                }
                //dsc1.modifyDatasetS(nARules1, startPerc, endPerc, oldIndexRR1, endIndexRR1, minCovElements1[0],maxCovElements1[0], rr1,appset.outFolderPath+"\\Jinputnew1.arff"/*"C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\UNCTADall3testnew1.arff"*/,fid);
            else if(appset.treeTypes.get(1)==0/*appset.typeOfRSTrees==0*/)
                if(appset.treeTypes.get(1)==1)
                     dsc1.modifyDatasetCat(startIndexRR1,endTmp, rr1,appset.outFolderPath+"\\Jinputnew1.arff",fid,appset);
                else
                     dsc1.modifyDatasetCat(startIndexRR1,endTmp, rr1,appset.outFolderPath+"/Jinputnew1.arff",fid,appset);
                //dsc1.modifyDatasetCat(nARules1, startPerc, endPerc, oldIndexRR1, endIndexRR1, minCovElements1[0],maxCovElements1[0], rr1,appset.outFolderPath+"\\Jinputnew1.arff"/*"C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\UNCTADall3testnew1.arff"*/,fid);
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
                    dsc1.modifyDatasetS(startIndexRR,endTmp,rr,appset.outFolderPath+"\\Jinputnew1.arff",fid,appset);
                 else
                    dsc1.modifyDatasetS(startIndexRR,endTmp,rr,appset.outFolderPath+"/Jinputnew1.arff",fid,appset); 
                //dsc1.modifyDatasetS(nARules1, startPerc, endPerc, oldIndexRR, endIndexRR, minCovElements[0],maxCovElements[0] ,rr,appset.outFolderPath+"\\Jinputnew1.arff"/*"C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\UNCTADall3testnew1.arff"*/,fid);
             else if(appset.treeTypes.get(0)==0/*appset.typeOfRSTrees==0*/)
                 if(appset.system.equals("windows"))
                 dsc1.modifyDatasetCat(startIndexRR,endTmp,rr,appset.outFolderPath+"\\Jinputnew1.arff",fid,appset);
             else
                   dsc1.modifyDatasetCat(startIndexRR,endTmp,rr,appset.outFolderPath+"/Jinputnew1.arff",fid,appset);   
                //dsc1.modifyDatasetCat(nARules1, startPerc, endPerc, oldIndexRR, endIndexRR, minCovElements[0],maxCovElements[0] ,rr,appset.outFolderPath+"\\Jinputnew1.arff"/*"C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\UNCTADall3testnew1.arff"*/,fid);
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
             nclMat.writeToFile(new File(appset.outFolderPath+"\\distances.csv"), fid,appset);
            }
           else{
              nclMat.resetFile(new File(appset.outFolderPath+"/distances.csv"));
             nclMat.writeToFile(new File(appset.outFolderPath+"/distances.csv"), fid,appset); 
           }
       }
       
        //run the second proces on new data
        // iterate until convergence (no new rules, or very small amount obtained)
         if(leftSide==1 && (endIndexRR1-oldIndexRR1)>z*appset.numTargets){
             exec.run(appset.javaPath, appset.clusPath, appset.outFolderPath, "view2tmp.s"/*"wbtmp.s"*/, 0,appset.clusteringMemory);//was 1 for rules before
             if(appset.numSupplementTrees>0)
                exec.run(appset.javaPath, appset.clusPath, appset.outFolderPath, "view2tmpF.s"/*"wbtmp.s"*/, 0,appset.clusteringMemory);//was 1 for rules before
             System.out.println("Process 2 side 1 finished!");
         }
         else if(rightSide==1 && (endIndexRR-oldIndexRR)>z*appset.numTargets){
             exec.run(appset.javaPath, appset.clusPath, appset.outFolderPath, "view1tmp.s"/*"unctadtmp.s"*/, 0,appset.clusteringMemory);
             if(appset.numSupplementTrees>0)
                 exec.run(appset.javaPath, appset.clusPath, appset.outFolderPath, "view1tmpF.s"/*"unctadtmp.s"*/, 0,appset.clusteringMemory);
             System.out.println("Process 1 side 1 finished!");
         }

        if((appset.useNC.get(0)==true && rightSide1==1 && appset.networkInit==false && appset.useNetworkAsBackground==false) || (appset.useNC.get(1)==true && leftSide1==1 && appset.networkInit==false && appset.useNetworkAsBackground==false)){ 
              if(appset.system.equals("windows")){ 
                 nclMat1.resetFile(new File(appset.outFolderPath+"\\distances.csv"));
                 nclMat1.writeToFile(new File(appset.outFolderPath+"\\distances.csv"), fid,appset);
              }
              else{
                 nclMat1.resetFile(new File(appset.outFolderPath+"/distances.csv"));
                 nclMat1.writeToFile(new File(appset.outFolderPath+"/distances.csv"), fid,appset);
              }
        }
         
         //run the second proces for cicle 2 on new data
        // iterate until convergence (no new rules, or very small amount obtained)
         if(leftSide1==1 && (endIndexRR1-oldIndexRR1)>z*appset.numTargets){
             exec.run(appset.javaPath, appset.clusPath, appset.outFolderPath,"view2tmp1.s" /*"wbtmpS1.s"*/, 0,appset.clusteringMemory);
             if(appset.numSupplementTrees>0)
                exec.run(appset.javaPath, appset.clusPath, appset.outFolderPath,"view2tmpF1.s" /*"wbtmpS1.s"*/, 0,appset.clusteringMemory);
             System.out.println("Process 2 side 2 finished!");
         }
         else if(rightSide1==1 && (endIndexRR-oldIndexRR)>z*appset.numTargets){
             exec.run(appset.javaPath, appset.clusPath, appset.outFolderPath,"view1tmp1.s"/*"unctadtmpS1.s"*/ , 0,appset.clusteringMemory);
             if(appset.numSupplementTrees>0)
             exec.run(appset.javaPath, appset.clusPath, appset.outFolderPath,"view1tmpF1.s"/*"unctadtmpS1.s"*/ , 0,appset.clusteringMemory);
             System.out.println("Process 1 side 2 finished!");
         }

       //extract rules for cicle 1
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
        //String input1="C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\unctad.out";
       int newRules=0;
       if((leftSide==1 && (endIndexRR1-oldIndexRR1)>z*appset.numTargets) || (rightSide==1 && (endIndexRR-oldIndexRR)>z*appset.numTargets)){
            ItRules.extractRules(input,fid,datJ,appset);
            ItRules.setSize();
            if(appset.numSupplementTrees>0){
                ItRulesF.extractRules(inputF,fid,datJ,appset);
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
       // rr.extractRules(input);
        }
        else if(rightSide==1 && (endIndexRR-oldIndexRR)>z*appset.numTargets){
            if(z==0)
                newRules=rr1.addnewRulesC(ItRules, appset.numnewRAttr,1);
            else
                newRules=rr1.addnewRulesC(ItRules, appset.numnewRAttr, 0);
            if(appset.numSupplementTrees>0)
                rr1.addnewRulesCF(ItRulesF, appset.numnewRAttr); 
            //rr1.extractRules(input);
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
        //String input1="C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\unctad.out";
       int newRules1=0;
       if((leftSide1==1 && (endIndexRR1-oldIndexRR1)>z*appset.numTargets) || (rightSide1==1 && (endIndexRR-oldIndexRR)>z*appset.numTargets)){
        ItRules1.extractRules(input,fid,datJ,appset);
        ItRules1.setSize();
        if(appset.numSupplementTrees>0){
             ItRulesF1.extractRules(inputF,fid,datJ,appset);
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
       // rr.extractRules(input);
        }
        else if(rightSide1==1 && (endIndexRR-oldIndexRR)>z*appset.numTargets){
            if(z==0)
                    newRules1=rr1.addnewRulesC(ItRules1, appset.numnewRAttr,1);
            else
                    newRules1=rr1.addnewRulesC(ItRules1, appset.numnewRAttr,0);
            if(appset.numSupplementTrees>0)
             rr1.addnewRulesCF(ItRulesF1, appset.numnewRAttr);
            //rr1.extractRules(input);
        }

        System.out.println("New rules cicle 1: "+newRules);
        System.out.println("New rules cicle 2: "+newRules1);
       }
        //add the redescription creaton code
       
       if(appset.optimizationType==0){ 
        if(appset.useJoin){
            //add computation of rule support if bagging
            newRedescriptions=rs.createGuidedJoinBasic(rr1, rr, jsN, appset, oldIndexRR1, oldIndexRR, RunInd,oom,fid,datJ, elemFreq, attrFreq, redScores,redScoresAtt,redDistCoverage,redDistCoverageAt, redDistNetwork, targetAtScore, Statistics, maxDiffScoreDistribution,nclMatInit,0);
           // rr.removeElements(rr.newRuleIndex);
            //rr1.removeElements(rr1.newRuleIndex);
            if(appset.numSupplementTrees>0){
                 rr.removeRulesCF();
                 rr1.removeRulesCF();
            }
        }
        else if(!appset.useJoin){
            newRedescriptions=rs.createGuidedNoJoinBasic(rr1, rr, jsN, appset, oldIndexRR1, oldIndexRR, RunInd,oom,fid,datJ, elemFreq, attrFreq, redScores,redScoresAtt,redDistCoverage,redDistCoverageAt, redDistNetwork, targetAtScore, Statistics, maxDiffScoreDistribution,nclMatInit,0);
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
            newRedescriptions=rs.createGuidedJoinExt(rr1, rr, jsN, appset, oldIndexRR1, oldIndexRR, RunInd, oom,fid,datJ);//sqitch sides of rules
            //rr.removeElements(rr.newRuleIndex);
            //rr1.removeElements(rr1.newRuleIndex);
        }
        else if(!appset.useJoin){
            newRedescriptions=rs.createGuidedNoJoinExt(rr1, rr, jsN, appset, oldIndexRR1, oldIndexRR, RunInd,oom,fid,datJ);
            rr.removeElements(rr.newRuleIndex);
            rr1.removeElements(rr1.newRuleIndex);
        }
       }
       
      /* if(rs.redescriptions.size()==2)
           return;*/

         it++;
        
        System.out.println("New redescriptions: "+newRedescriptions);
         
        //if more than two viewes get guided search in here for further views...
        //should be modified for redescription addition
        //new join procedures should be created 
        System.out.println("Number of viewes: "+datJ.W2indexs.size());
        for(int nws=2;nws<datJ.W2indexs.size()+1;nws++){
            if(readers.size()<(datJ.W2indexs.size()-2)+1)
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
            if((nws-1)<(datJ.W2indexs.size()-2+1))
                setMW.createInitialSettingsGen(nws, datJ.W2indexs.get(nws-1)+1 ,datJ.W2indexs.get(nws)+1,datJ.schema.getNbAttributes() , appset,0);
            else
                setMW.createInitialSettingsGen(nws, datJ.W2indexs.get(nws-1)+1 ,datJ.schema.getNbAttributes()+1,datJ.schema.getNbAttributes() , appset,0);
            
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
                             nclMat.loadDistance(new File(appset.distanceFilePaths.get(nws)), fid);
                              if(appset.system.equals("windows")){ 
                                     nclMat.writeToFile(new File(appset.outFolderPath+"\\distance.csv"), fid,appset);
                              }
                              else{
                                  nclMat.writeToFile(new File(appset.outFolderPath+"/distance.csv"), fid,appset);
                              }
                    }
                     else if(appset.computeDMfromRules==true){
                             nclMat.computeDistanceMatrix(rs.redescriptions, fid, appset.maxDistance, datJ.numExamples,oldRIndex);
                             if(appset.system.equals("windows")){ 
                                    nclMat.resetFile(new File(appset.outFolderPath+"\\distances.csv"));
                                    nclMat.writeToFile(new File(appset.outFolderPath+"\\distances.csv"), fid,appset);
                             }
                             else{
                                 nclMat.resetFile(new File(appset.outFolderPath+"/distances.csv"));
                                 nclMat.writeToFile(new File(appset.outFolderPath+"/distances.csv"), fid,appset);
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
            
           /* cuttof=rs.findCutoff(naex, startPerc, endPerc, minCovElements,maxCovElements,oldRIndex, appset.minSupport,appset.maxSupport,appset.numTargets);
             System.out.println("minCovElements: "+minCovElements[0]);
             System.out.println("maxCovElements: "+maxCovElements[0]);
             System.out.println("cuttof: "+cuttof);
             
           if(cuttof==-1)
                continue;*/
            
           if(appset.system.equals("windows")) 
                dsc=new DataSetCreator(appset.outFolderPath+"\\Jinput.arff");
           else
               dsc=new DataSetCreator(appset.outFolderPath+"/Jinput.arff");
            
             try{
        dsc.readDataset();
        }
        catch(IOException e){
            e.printStackTrace();
        }
            
            System.out.println("startPerc: "+startPerc);
            System.out.println("endPerc: "+endPerc);
            
            int endTmp=0;
             if((z+1)*appset.numTargets>rs.redescriptions.size())
                 endTmp=rs.redescriptions.size();
             else endTmp=(z+1)*appset.numTargets;
              //add conditions in the this part of the code...!!!
             int startIndexRR=oldRIndex[0]+z*appset.numTargets;
            
            for(int i=startIndexRR;i<endTmp;i++)//oldRIndex[0];i<rs.redescriptions.size();i++) //do on the fly when reading rules
                   // if(rs.redescriptions.get(i).elements.size()<=naex*endPerc && rs.redescriptions.get(i).elements.size()>=naex*startPerc && rs.redescriptions.get(i).elements.size()>=minCovElements[0] && rs.redescriptions.get(i).elements.size()<=maxCovElements[0]) //do parameters analysis in this step
                        nARules++;
             setMW.ModifySettings(nARules,dsc.schema.getNbAttributes());
             try{
                 if(appset.treeTypes.get(nws)==1/*appset.typeOfRSTrees==1*/){ 
                     if(appset.system.equals("windows")) 
                         dsc.modifyDatasetS(startIndexRR,endTmp, rs.redescriptions,appset.outFolderPath+"\\Jinputnew.arff",fid,appset);
                     else 
                         dsc.modifyDatasetS(startIndexRR,endTmp, rs.redescriptions,appset.outFolderPath+"/Jinputnew.arff",fid,appset);
                 }
         else if(appset.treeTypes.get(nws)==0/*appset.typeOfRSTrees==0*/){
             if(appset.system.equals("windows")) 
                dsc.modifyDatasetCat(startIndexRR,endTmp, rs.redescriptions,appset.outFolderPath+"\\Jinputnew.arff",fid,appset);
             else
                dsc.modifyDatasetCat(startIndexRR,endTmp, rs.redescriptions,appset.outFolderPath+"/Jinputnew.arff",fid,appset); 
         }
         //if(appset.treeTypes.get(nws)==1/*appset.typeOfRSTrees==1*/)   
               // dsc.modifyDatasetS(nARules, startPerc, endPerc, oldRIndex[0], rs.redescriptions.size(), minCovElements[0],maxCovElements[0], rs.redescriptions,appset.outFolderPath+"\\Jinputnew.arff",fid);
        // else if(appset.treeTypes.get(nws)==0/*appset.typeOfRSTrees==0*/)
          //      dsc.modifyDatasetCat(nARules, startPerc, endPerc, oldRIndex[0], rs.redescriptions.size(), minCovElements[0],maxCovElements[0], rs.redescriptions,appset.outFolderPath+"\\Jinputnew.arff",fid);
        }
        catch(IOException e){
            e.printStackTrace();
        }
             
             exec.run(appset.javaPath, appset.clusPath, appset.outFolderPath, "view3tmp.s"/*"wbtmp.s"*/, 0,appset.clusteringMemory);//was 1 for rules before
             System.out.println("Process 1 side "+nws+" finished!");
             
             String input;
             if(appset.system.equals("windows")) 
              input=appset.outFolderPath+"\\view3tmp.out";
             else
                 input=appset.outFolderPath+"/view3tmp.out";
             
              int newRules=0;
              RuleReader ItRules=new RuleReader();
              
             ItRules.extractRules(input,fid,datJ,appset);
        ItRules.setSize();
            if(z==0)
                newRules=readers.get(nws-2).addnewRulesC(ItRules, appset.numnewRAttr,1);
            else
                newRules=readers.get(nws-2).addnewRulesC(ItRules, appset.numnewRAttr,0);
           }
        
        if(appset.useJoin){//do redescription construction
            rs.combineViewRulesJoin(readers.get(nws-2), jsN, appset, oldIndW, RunInd, oom, fid, datJ, oldRIndex ,nws); 
        }
        else{//(rr, rr1, jsN, appset, oldIndexRR, oldIndexRR1, RunInd, oom,fid,datJ);
            rs.combineViewRules(readers.get(nws-2), jsN, appset, oldIndW, RunInd, oom, fid, datJ, oldRIndex ,nws);
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
        System.out.println("Running index: "+RunInd);
        }
      }
       
        System.out.println("Redescription size main: "+rs.redescriptions.size());
        
        //removing all redescriptions with inadequate minSupport and minJS
        rs.remove(appset);
         
        System.out.println("Redescription size main after remove: "+rs.redescriptions.size());
        
        //filtering
      // rs.filter(appset, rr, rr1,fid,datJ); // think about what we want and if we need it
        
        System.out.println("Redescription size main after filter: "+rs.redescriptions.size());
       
      int numFullRed=0;
        //computing pVal...
        numFullRed=rs.computePVal(datJ,fid);
        rs.removePVal(appset);
        
         System.out.println("Redescription size main after Pvalremove: "+rs.redescriptions.size());
        
        System.out.println("Found "+numFullRed+" redescriptions with JS=1.0 and minsupport>"+appset.minSupport);
        System.out.println("Found "+rs.redescriptions.size()+" redescriptions with JS>"+appset.minJS);
        
         int minimize=0;
         if(appset.minimizeRules==true)
             minimize=1;
        
         rs.adaptSet(datJ, fid,0);
          System.out.println(rs.redescriptions.size()+" redescriptions with JS>"+appset.minJS+" left after elimination of long rules");
        //further reduce before computing optimized set

         /*if(numFullRed>appset.numRetRed){
            appset.minJS=1.0;
            rs.remove(appset);
        }
        else if(numFullRed<appset.numRetRed && rs.redescriptions.size()>appset.numRetRed){
            while(true){
                int num=rs.countNumber(appset.minJS+0.1);
                if((rs.redescriptions.size()-num)>appset.numRetRed){
                    appset.minJS=appset.minJS+0.1;
                    rs.remove(appset);
                }
                else break;
            }
        }*/ //removed for the time being
        
        //sorting redescriptions
        System.out.println("Sorting rules!");
        rs.sortRedescriptions();
       
        if(appset.sameView==1)
        for(int k=rs.redescriptions.size()-1;k>=0;k--)
            if(rs.redescriptions.get(k).disj(datJ, fid) == 0)
                rs.redescriptions.remove(k);
        
       /* System.out.println("Validation in main");
         //this.adaptSet(dat, map);
          for(int i=0;i<rs.redescriptions.size();i++)//uncomment
            rs.redescriptions.get(i).removeRedundant();*/
  
         for(int i=0;i<rs.redescriptions.size();i++){
             //rs.redescriptions.get(i).closeInterval(datJ, fid);
            // rs.redescriptions.get(i).validate(datJ, fid);
             if(appset.classHomogeneity == true)
             rs.redescriptions.get(i).setClassCount(targets, targetsToIndex, fid);
         }
         
          if(appset.useSplitTesting==true)
              for(int i=0;i<rs.redescriptions.size();i++){
                     rs.redescriptions.get(i).ComputeValidationStatistics(datJ,datJTrainValid, fid);//ovdje koristiti train+validation???
                     rs.redescriptions.get(i).ComputeTestStatistics(datJ,datJValid, fid);
              }
         
         
         for(int i=0;i<rs.redescriptions.size();i++)
             rs.redescriptions.get(i).clearRuleMaps();
         
        if(appset.attributeImportance==0) 
         rs.adaptSet(datJ, fid,minimize);
        else
            rs.adaptSet(datJ, fid, 0);
         
        /* CoocurenceMatrix coc=new CoocurenceMatrix(datJ.numExamples,datJ.schema.getNbAttributes()-1);
         coc.computeMatrix(rs, datJ);
         File out=new File(appset.outFolderPath+"\\Elements.txt");
         coc.writeToFileElements(out, datJ.numExamples);
         out=new File(appset.outFolderPath+"\\Attributes.txt");
         coc.writeToFileAttributes(out,datJ.schema.getNbAttributes()-1);*/
         
       System.out.println("Computing rule score and sorting!");
      RedescriptionSet Result=rs;//new RedescriptionSet();
      
      if(appset.optimizationType==0){
      
      double sumN=0.0;
     /* if(appset.JSImpWeight+appset.PValImpWeight+appset.AttDivImpWeight+appset.ElemDivImpWeight+appset.RuleSizeImpWeight>1.0){
            sumN=appset.JSImpWeight+appset.PValImpWeight+appset.AttDivImpWeight+appset.ElemDivImpWeight+appset.RuleSizeImpWeight;
            appset.JSImpWeight/=sumN; appset.PValImpWeight/=sumN; appset.AttDivImpWeight/=sumN; appset.ElemDivImpWeight/=sumN; appset.RuleSizeImpWeight/=sumN;
      }*/
      
      double heuristicWeights[]=appset.preferences.get(0);
      //double heuristicWeights[]=new double[]{appset.JSImpWeight,appset.PValImpWeight,appset.ElemDivImpWeight,appset.AttDivImpWeight,appset.RuleSizeImpWeight};
     // Result.createRedescriptionSetCooc(rs, heuristicWeights, appset,datJ,fid,coc);
      //Result.createRedescriptionSet(rs, heuristicWeights, appset,datJ,fid);
      double coverage[]=new double[2];
      double ResultsScore=Result.computeRedescriptionSetScore(heuristicWeights,coverage,datJ,fid);
      System.out.println("Results score: "+ResultsScore);
        //writing redescriptions to file
      //rs.writeToFile(appset.outFolderPath+"\\"+appset.outputName, datJ, fid, rr, rr1, startTime,numFullRed,appset);//fix output file name
     if(appset.system.equals("windows")){ 
      Result.writeToFile(appset.outFolderPath+"\\"+appset.outputName+(1)+".rr", datJ, fid, startTime,numFullRed,appset, ResultsScore, coverage,oom);
      Result.writePlots(appset.outFolderPath+"\\"+"RuleData"+(1)+".csv", appset,datJ,fid);
     }
     else{
       Result.writeToFile(appset.outFolderPath+"/"+appset.outputName+(1)+".rr", datJ, fid, startTime,numFullRed,appset, ResultsScore, coverage,oom);
      //Result.writePlots(appset.outFolderPath+"/"+"RuleData"+(1)+".csv", appset,datJ,fid);  
     }
      }
      else{
          double coverage[];
           rs.computeAllMeasureFS(datJ, appset, fid);
         
      double ResultsScore=0.0;
      
         CoocurenceMatrix coc=null;
         
         if(datJ.numExamples<10000 && datJ.schema.getNbAttributes()-1<10000){
                coc=new CoocurenceMatrix(datJ.numExamples,datJ.schema.getNbAttributes()-1);
                coc.computeMatrix(rs, datJ); 
         }
         
       Result=new RedescriptionSet();
      
      double sumN=0.0;
      /*if(appset.JSImpWeight+appset.PValImpWeight+appset.AttDivImpWeight+appset.ElemDivImpWeight+appset.RuleSizeImpWeight>1.0){
            sumN=appset.JSImpWeight+appset.PValImpWeight+appset.AttDivImpWeight+appset.ElemDivImpWeight+appset.RuleSizeImpWeight;
            appset.JSImpWeight/=sumN; appset.PValImpWeight/=sumN; appset.AttDivImpWeight/=sumN; appset.ElemDivImpWeight/=sumN; appset.RuleSizeImpWeight/=sumN;
      }*/
      
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
      if(datJ.numExamples<10000 && datJ.schema.getNbAttributes()-1<10000)
          if(appset.classHomogeneity == true)
            resSets=Result.createRedescriptionSetsCoocGenPred(rs,appset.preferences,appset.parameters.get(i).get(2).intValue(), appset,datJ,fid,coc);//adds the most specific redescription first
          else  resSets=Result.createRedescriptionSetsCoocGen(rs,appset.preferences,appset.parameters.get(i).get(2).intValue(), appset,datJ,fid,coc);//adds the most specific redescription first
      else
          resSets=Result.createRedescriptionSetsRandGen(rs,appset.preferences,appset.parameters.get(i).get(2).intValue(), appset,datJ,fid,coc);//should add one highly accurate redescription at random
      
      if(resSets==null){//perhaps create a null file
          break;
      }

      for(int rset=0;rset<resSets.size();rset++)
            resSets.get(rset).computeLift(datJ, fid);
  
      System.out.println("exhaustiveTesting = 0");
      System.out.println("RS size: "+resSets.size());
      
     for(int fit=0;fit<resSets.size();fit++){
       coverage=new double[2];

      ResultsScore=resSets.get(fit).computeRedescriptionSetScoreGen(appset.preferences.get(fit),appset.parameters.get(i).get(2).intValue(),coverage,datJ,appset,fid);
      System.out.println("Results score: "+ResultsScore);
      numFullRed=resSets.get(fit).computePVal(datJ,fid);

      if(appset.system.equals("windows"))
         resSets.get(fit).writeToFile(appset.outFolderPath+"\\"+appset.outputName+"StLev_"+fit+" minjs "+appset.minJS+" JSType "+appset.parameters.get(i).get(2).intValue()+".rr", datJ, fid, startTime,numFullRed,appset, ResultsScore, coverage,oom);
      else
        resSets.get(fit).writeToFile(appset.outFolderPath+"/"+appset.outputName+"StLev_"+fit+" minjs "+appset.minJS+" JSType "+appset.parameters.get(i).get(2).intValue()+".rr", datJ, fid, startTime,numFullRed,appset, ResultsScore, coverage,oom);  
      
      if(appset.system.equals("windows"))
             resSets.get(fit).writePlots(appset.outFolderPath+"\\"+"RuleData"+"StLev_"+fit+" minjs "+appset.minJS+"JSType "+appset.parameters.get(i).get(2).intValue()+".csv", appset,datJ,fid);
      else
            resSets.get(fit).writePlots(appset.outFolderPath+"/"+"RuleData"+"StLev_"+fit+" minjs "+appset.minJS+"JSType "+appset.parameters.get(i).get(2).intValue()+".csv", appset,datJ,fid);

      predFinal = resSets.get(0);
      //coc.init(datJ.numExamples, datJ.schema.getNbAttributes()-1);
     
     //resSets.get(fit).redescriptions.clear();
     }
      }
      
   
   /*      Iterator<Integer> itC = fid.idAtt.keySet().iterator();
     
     while(itC.hasNext()){
         int atid = itC.next();
         
         String at = fid.idAtt.get(atid);
         String at1 = fidFull.idAtt.get(atid);
         
         System.out.println(at+" "+at1);
         
     }
     
     itC = fid.idExample.keySet().iterator();
     
     while(itC.hasNext()){
         int elid = itC.next();
         
         String el = fid.idExample.get(elid);
         String el1 = fidFull.idExample.get(elid);
         
         if(!el.equals(el1))
             System.out.println("Diff: "+el+" "+el1);
         
     }*/
     
      
      RuleReader rrAll = new RuleReader();
      //need to add elements sto rules
      for(int i=0;i<rr.rules.size();i++){
          rr.rules.get(i).closeInterval(datJ, fid);
         // System.out.println(rr.rules.get(i).rule);
          if(appset.classHomogeneity == true)
           rr.rules.get(i).setClassCount(targets, targetsToIndex, fid);
          rr.rules.get(i).addElements1(fidFull, fid ,datJFull);
         
      }
      
     rrAll.rules.addAll(rr.rules);

   
     for(int i=0;i<rr1.rules.size();i++){
         rr1.rules.get(i).closeInterval(datJ, fid);
        // System.out.println(rr1.rules.get(i).rule);
         if(appset.classHomogeneity == true)
         rr1.rules.get(i).setClassCount(targets, targetsToIndex, fid);
          rr1.rules.get(i).addElements1(fidFull,fid ,datJFull);          
      }
    
     rrAll.rules.addAll(rr1.rules);
     
     String ruleFile = appset.outFolderPath;
     
          rrAll.filterRules(0, 0.8, appset);//0.8 default//0.5 last value
          if(appset.classHomogeneity == true)
            rrAll.filterRulesClassPurity(0, 1.0 ,appset);// 0.6 default//0.8 last value
     
     if(appset.system.equals("windows"))
         ruleFile+="\\rules.rr";
     else ruleFile+="/rules.rr";
     
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
     
     System.out.println("rrAll size: "+rrAll.rules.size());
     
           
      //add here testing on orig features + rule featues only + feature selection
    
       if(appset.system.equals("windows")){
            dataTrainValidation = wdl.loadDataset(appset.outFolderPath+"\\TrainValidation.arff");
            dataTest=wdl.loadDataset(appset.outFolderPath+"\\Test.arff");
     }
     else{
        dataTrainValidation = wdl.loadDataset(appset.outFolderPath+"/TrainValidation.arff");
            dataTest=wdl.loadDataset(appset.outFolderPath+"/Test.arff"); 
     }
     
      dataTest.setClassIndex(dataTest.numAttributes()-1);
      dataTrainValidation.setClassIndex(dataTrainValidation.numAttributes()-1);
     
      data = wdl.loadDataset(appset.viewInputPaths.get(0));//load train/validation/test, than add to all
      data.setClassIndex(data.numAttributes()-1);  
      
      ArrayList<String> entityNames1=new ArrayList<>();
      
      for(int i=0;i<data.numInstances();i++){
          entityNames1.add(data.instance(i).stringValue(0));
      }
      
      int indO1 = data.numAttributes()-1;
      
      System.out.println("Num attrs old: "+indO1);
      
      for(int i=0;i<rrAll.rules.size();i++){
          data.insertAttributeAt(new Attribute("r"+i), data.numAttributes()-1);
          dataTest.insertAttributeAt(new Attribute("r"+i), dataTest.numAttributes()-1);
          dataTrainValidation.insertAttributeAt(new Attribute("r"+i), dataTrainValidation.numAttributes()-1);
      }

      
       String entN1="";
     
      int totNum1 = indO1;
      
     for(int j=0;j<rrAll.rules.size();j++) 
      for(int i=0;i<data.numInstances();i++){
          entN1 = data.instance(i).stringValue(0);
          if(rrAll.rules.get(j).elements == null){
              System.out.println("Rule with null elements: "+j+"\n "+rrAll.rules.get(j).rule);
          }
          if(!fidFull.exampleId.containsKey(entN1)){
              System.out.println("Not contained: "+entN1);
              if(fidFull.exampleId.containsKey("\"+entN+\""))
                  entN1 = "\""+entN1+"\"";
          }
          if(rrAll.rules.get(j).elements.contains(fidFull.exampleId.get(entN1)))
                    data.instance(i).setValue(indO1+j, 1);
          else  data.instance(i).setValue(indO1+j, 0);
      }

     
     String predFile1 = "";
     try{
      if(appset.system.equals("windows"))//write new file to disk
                        predFile1= appset.outFolderPath+"\\FeatureConstructionRules.arff";
                    else predFile1= appset.outFolderPath+"/FeatureConstructionRules.arff";
                     DataSink.write(predFile1, data);
     }
     catch(Exception e){
         e.printStackTrace();
     } //all data
     
      for(int j=0;j<rrAll.rules.size();j++) 
      for(int i=0;i<dataTrainValidation.numInstances();i++){
          entN1 = dataTrainValidation.instance(i).stringValue(0);
          if(rrAll.rules.get(j).elements.contains(fidFull.exampleId.get(entN1)))
                    dataTrainValidation.instance(i).setValue(indO1+j, 1);
          else  dataTrainValidation.instance(i).setValue(indO1+j, 0);
      }

   predFile1 = "";
     try{
      if(appset.system.equals("windows"))//write new file to disk
                        predFile1= appset.outFolderPath+"\\FeatureConstructionTrainValidationRules.arff";
                    else predFile1= appset.outFolderPath+"/FeatureConstructionTrainValidationRules.arff";
                     DataSink.write(predFile1, dataTrainValidation);
     }
     catch(Exception e){
         e.printStackTrace();
     }//train-validation data
     
     
      for(int j=0;j<rrAll.rules.size();j++) 
      for(int i=0;i<dataTest.numInstances();i++){
          entN1 = dataTest.instance(i).stringValue(0);
          if(rrAll.rules.get(j).elements.contains(fidFull.exampleId.get(entN1)))
                    dataTest.instance(i).setValue(indO1+j, 1);
          else  dataTest.instance(i).setValue(indO1+j, 0);
      }      
      
       predFile1 = "";
     try{
      if(appset.system.equals("windows"))//write new file to disk
                        predFile1= appset.outFolderPath+"\\FeatureConstructionTestRules.arff";
                    else predFile1= appset.outFolderPath+"/FeatureConstructionTestRules.arff";
                     DataSink.write(predFile1, dataTest); 
     }
     catch(Exception e){
         e.printStackTrace();
     }//test data
      
     
      //compute evaluation using all features
     SettingsReader predictiveSettings1 = new SettingsReader();
     
     String predTV1="",pTest1 = "";
     
     if(appset.system.equals("windows")){//write new file to disk
                        predTV1= appset.outFolderPath+"\\FeatureConstructionTrainValidationRules.arff";
                        pTest1 = appset.outFolderPath+"\\FeatureConstructionTestRules.arff";
     }
     else{ predTV1= appset.outFolderPath+"/FeatureConstructionTrainValidationRules.arff";
            pTest1= appset.outFolderPath+"/FeatureConstructionTestRules.arff";
     }
     
     
     if(appset.system.equals("windows")){//createPredictiveSettingsFunction
             predictiveSettings1.setPath(appset.outFolderPath+"\\predictiveAllRules.s");//replace view1 with real name
                 predictiveSettings1.createPredictiveSettings(predTV1, pTest1, dataTrainValidation.numAttributes(), 1, 600, 0);

     }
     else{
             predictiveSettings1.setPath(appset.outFolderPath+"/predictiveAllRules.s");
                 predictiveSettings1.createPredictiveSettings(predTV1, pTest1, dataTrainValidation.numAttributes(),1 , 600, 0);
     }
     
     //call CLUS
      ClusProcessExecutor exec=new ClusProcessExecutor();
     
      exec.run(appset.javaPath,appset.clusPath ,appset.outFolderPath,"predictiveAllRules.s",0, appset.clusteringMemory);

     //call clus settings and call CLUS
     //create weka evaluation
      
      Instances dataTrainValidationAll = null;
      Instances dataTestAll = null;
      
       if(appset.system.equals("windows")){
                dataTrainValidationAll = wdl.loadDataset(appset.outFolderPath+"\\FeatureConstructionTrainValidationRules.arff");
                dataTestAll = wdl.loadDataset(appset.outFolderPath+"\\FeatureConstructionTestRules.arff");
                dataTestAll.setClassIndex(dataTest.numAttributes()-1);
                dataTrainValidationAll.setClassIndex(dataTrainValidation.numAttributes()-1);
           }
           else{
                dataTrainValidationAll = wdl.loadDataset(appset.outFolderPath+"/FeatureConstructionTrainValidationRules.arff");
                dataTestAll = wdl.loadDataset(appset.outFolderPath+"/FeatureConstructionTestRules.arff");
                dataTestAll.setClassIndex(dataTestAll.numAttributes()-1);
                dataTrainValidationAll.setClassIndex(dataTrainValidationAll.numAttributes()-1);
           }
       
       HashMap<String,ArrayList<Double>> resultsWeka = new HashMap<>();
       Classifier clsf = new J48();
      dataTrainValidationAll.deleteAttributeAt(0);
      dataTestAll.deleteAttributeAt(0);
      try{
            clsf.buildClassifier(dataTrainValidationAll);
            Evaluation eval = new Evaluation(dataTrainValidationAll);
            eval.evaluateModel(clsf, dataTestAll); 
            double auprc = 0, auc = 0;
            for(int i=0;i<dataTrainValidationAll.classAttribute().numValues();i++){
                System.out.println("AUPRC"+i+": "+eval.areaUnderPRC(i));
                System.out.println("AUC"+i+": "+eval.areaUnderROC(i));
                auprc+=eval.areaUnderPRC(i); auc+=eval.areaUnderROC(i);
            }
            auprc/=dataTrainValidationAll.classAttribute().numValues();
            auc/=dataTrainValidationAll.classAttribute().numValues();
            /*System.out.println("AUPRC0: "+eval.areaUnderPRC(0));
            System.out.println("AUC0: "+eval.areaUnderROC(0));
            System.out.println("AUPRC1: "+eval.areaUnderPRC(1));
            System.out.println("AUC1: "+eval.areaUnderROC(1));*/
            System.out.println("Av.AUPRC J48: "+auprc);
            System.out.println("Av.AUC J48: "+auc);
            System.out.println(eval.toSummaryString("\nResults FCAllRules\n=====\n",false));
            
            if(!resultsWeka.containsKey("J48")){
                resultsWeka.put("J48", new ArrayList<>());
            }
            
            resultsWeka.get("J48").add(auprc);
             resultsWeka.get("J48").add(auc);
            
            clsf = new NaiveBayes();
            
             clsf.buildClassifier(dataTrainValidationAll);
            eval = new Evaluation(dataTrainValidationAll);
            eval.evaluateModel(clsf, dataTestAll);
             auprc = 0; auc = 0;
            for(int i=0;i<dataTrainValidationAll.classAttribute().numValues();i++){
                System.out.println("AUPRC"+i+": "+eval.areaUnderPRC(i));
                System.out.println("AUC"+i+": "+eval.areaUnderROC(i));
                auprc+=eval.areaUnderPRC(i); auc+=eval.areaUnderROC(i);
            }
            auprc/=dataTrainValidationAll.classAttribute().numValues();
            auc/=dataTrainValidationAll.classAttribute().numValues();
            System.out.println("Av.AUPRC NB: "+auprc);
            System.out.println("Av.AUC NB: "+auc);
            System.out.println(eval.toSummaryString("\nResults FCAllRules\n=====\n",false));
            
             if(!resultsWeka.containsKey("NB")){
                resultsWeka.put("NB", new ArrayList<>());
            }
            
            resultsWeka.get("NB").add(auprc);
             resultsWeka.get("NB").add(auc);
            
            clsf = new Logistic();
            
             clsf.buildClassifier(dataTrainValidationAll);
            eval = new Evaluation(dataTrainValidationAll);
            eval.evaluateModel(clsf, dataTestAll);
             auprc = 0; auc = 0;
            for(int i=0;i<dataTrainValidationAll.classAttribute().numValues();i++){
                System.out.println("AUPRC"+i+": "+eval.areaUnderPRC(i));
                System.out.println("AUC"+i+": "+eval.areaUnderROC(i));
                auprc+=eval.areaUnderPRC(i); auc+=eval.areaUnderROC(i);
            }
            auprc/=dataTrainValidationAll.classAttribute().numValues();
            auc/=dataTrainValidationAll.classAttribute().numValues();
            System.out.println("Av.AUPRC Log: "+auprc);
            System.out.println("Av.AUC Log: "+auc);
            System.out.println(eval.toSummaryString("\nResults FCAllRules\n=====\n",false));
            
            
             if(!resultsWeka.containsKey("Log")){
                resultsWeka.put("Log", new ArrayList<>());
            }
            
            resultsWeka.get("Log").add(auprc);
             resultsWeka.get("Log").add(auc);
            
            clsf = new MultilayerPerceptron();
            
             clsf.buildClassifier(dataTrainValidationAll);
            eval = new Evaluation(dataTrainValidationAll);
            eval.evaluateModel(clsf, dataTestAll);
             auprc = 0; auc = 0;
            for(int i=0;i<dataTrainValidationAll.classAttribute().numValues();i++){
                System.out.println("AUPRC"+i+": "+eval.areaUnderPRC(i));
                System.out.println("AUC"+i+": "+eval.areaUnderROC(i));
                auprc+=eval.areaUnderPRC(i); auc+=eval.areaUnderROC(i);
            }
            auprc/=dataTrainValidationAll.classAttribute().numValues();
            auc/=dataTrainValidationAll.classAttribute().numValues();
            System.out.println("Av.AUPRC MLP: "+auprc);
            System.out.println("Av.AUC MLP: "+auc);
            System.out.println(eval.toSummaryString("\nResults FCAllRules\n=====\n",false));
            
             if(!resultsWeka.containsKey("MLP")){
                resultsWeka.put("MLP", new ArrayList<>());
            }
            
            resultsWeka.get("MLP").add(auprc);
             resultsWeka.get("MLP").add(auc);
            
            clsf = new KStar(); 
            
             clsf.buildClassifier(dataTrainValidationAll);
            eval = new Evaluation(dataTrainValidationAll);
            eval.evaluateModel(clsf, dataTestAll);
             auprc = 0; auc = 0;
            for(int i=0;i<dataTrainValidationAll.classAttribute().numValues();i++){
                System.out.println("AUPRC"+i+": "+eval.areaUnderPRC(i));
                System.out.println("AUC"+i+": "+eval.areaUnderROC(i));
                auprc+=eval.areaUnderPRC(i); auc+=eval.areaUnderROC(i);
            }
            auprc/=dataTrainValidationAll.classAttribute().numValues();
            auc/=dataTrainValidationAll.classAttribute().numValues();
            System.out.println("Av.AUPRC KStar: "+auprc);
            System.out.println("Av.AUC KStar: "+auc);
            System.out.println(eval.toSummaryString("\nResults FCAllRules\n=====\n",false));
            
            
             if(!resultsWeka.containsKey("KS")){
                resultsWeka.put("KS", new ArrayList<>());
            }
            
            resultsWeka.get("KS").add(auprc);
             resultsWeka.get("KS").add(auc);
            
            clsf = new DecisionStump();
            
             clsf.buildClassifier(dataTrainValidationAll);
            eval = new Evaluation(dataTrainValidationAll);
            eval.evaluateModel(clsf, dataTestAll);
             auprc = 0; auc = 0;
            for(int i=0;i<dataTrainValidationAll.classAttribute().numValues();i++){
                System.out.println("AUPRC"+i+": "+eval.areaUnderPRC(i));
                System.out.println("AUC"+i+": "+eval.areaUnderROC(i));
                auprc+=eval.areaUnderPRC(i); auc+=eval.areaUnderROC(i);
            }
            auprc/=dataTrainValidationAll.classAttribute().numValues();
            auc/=dataTrainValidationAll.classAttribute().numValues();
            System.out.println("Av.AUPRC DStump: "+auprc);
            System.out.println("Av.AUC DStump: "+auc);
            System.out.println(eval.toSummaryString("\nResults FCAllRules\n=====\n",false));
            
             if(!resultsWeka.containsKey("DSt")){
                resultsWeka.put("DSt", new ArrayList<>());
            }
            
            resultsWeka.get("DSt").add(auprc);
             resultsWeka.get("DSt").add(auc);
            
            clsf = new LMT();
            
             clsf.buildClassifier(dataTrainValidationAll);
            eval = new Evaluation(dataTrainValidationAll);
            eval.evaluateModel(clsf, dataTestAll);
             auprc = 0; auc = 0;
            for(int i=0;i<dataTrainValidationAll.classAttribute().numValues();i++){
                System.out.println("AUPRC"+i+": "+eval.areaUnderPRC(i));
                System.out.println("AUC"+i+": "+eval.areaUnderROC(i));
                auprc+=eval.areaUnderPRC(i); auc+=eval.areaUnderROC(i);
            }
            auprc/=dataTrainValidationAll.classAttribute().numValues();
            auc/=dataTrainValidationAll.classAttribute().numValues();
            System.out.println("Av.AUPRC LMT: "+auprc);
            System.out.println("Av.AUC LMT: "+auc);
            System.out.println(eval.toSummaryString("\nResults FCAllRules\n=====\n",false));
            
             if(!resultsWeka.containsKey("LMT")){
                resultsWeka.put("LMT", new ArrayList<>());
            }
            
            resultsWeka.get("LMT").add(auprc);
             resultsWeka.get("LMT").add(auc);
      }
      catch(Exception e){
          e.printStackTrace();
      }

      //add feature selection and eval on orig features + rule features selected attrs
     
           HashMap<Integer,Double> evalM1 = new HashMap<>();//hashmap containing numFeatures + average score = (convex combination of AUC and AUPRC) 
     TreeMap<Integer,String> featureRankList1 = new TreeMap<>();//rank list sorted by feature importance (first - best, last-worst)
     HashMap<String,Integer> featInvRank1 = new HashMap<>();
     HashMap<String,Integer> attributeIndex1 = new HashMap<>();//map containing index of each attribute in the dataset
       HashMap<Integer,String> indexAttribute1 = new HashMap<>();
     String outputFilenameTrain1 = "", outputFilenameValidation1="";
       exec=new ClusProcessExecutor();
     try{
    for(int numTries = 0; numTries<20; numTries++){ //20 random splits to train/validation
        System.out.println("Feature selection iteration: "+(numTries+1)); 
        featureRankList1.clear(); featInvRank1.clear();
         dataTrainValidation.randomize(new java.util.Random(numTries));//create different split in each iteration
         
         insfilt = new InstancesFilter(dataTrainValidation);
                     insfilt.removeStratifiedFoldsFilter(1,4,true);
                     dataValidation = insfilt.getFilteredInstances();
                     
                     insfilt = new InstancesFilter(dataTrainValidation);
                     insfilt.removeStratifiedFoldsFilter(1,4,false);
                     dataTrain = insfilt.getFilteredInstances();
                     
                      if(appset.system.equals("windows"))
                                outputFilenameValidation1 = appset.outFolderPath+"\\ValidationAA.arff";
                     else  outputFilenameValidation1 = appset.outFolderPath+"/ValidationAA.arff";//save all attrs to file
                     DataSink.write(outputFilenameValidation1, dataValidation);
                     
                      if(appset.system.equals("windows"))
                             outputFilenameTrain1 = appset.outFolderPath+"\\TrainAA.arff";
                      else  outputFilenameTrain1 = appset.outFolderPath+"/TrainAA.arff";
                     DataSink.write(outputFilenameTrain1, dataTrain);
        
        int numAttributes = dataTrainValidation.numAttributes();
        //split to train and validation
        //save to disk
     int init=0;
        while(numAttributes>10){
            System.out.println("numAttributes: "+numAttributes);
            System.out.println("naTrain: "+dataTrain.numAttributes());
     //create predictive settings
      predictiveSettings1 = new SettingsReader();
     numAttributes = dataTrain.numAttributes();
     //load attribte index
     attributeIndex1.clear();
     indexAttribute1.clear();
     
     for(int z=0;z<dataTrain.numAttributes();z++){
         indexAttribute1.put(z, dataTrain.attribute(z).name());
         attributeIndex1.put(dataTrain.attribute(z).name(), z);
     }
     
     
     if(appset.system.equals("windows")){//createPredictiveSettingsFunction
             predictiveSettings1.setPath(appset.outFolderPath+"\\predictive.s");//replace view1 with real name
             if(init == 0){
                 predictiveSettings1.createPredictiveSettings(outputFilenameTrain1, outputFilenameValidation1, numAttributes, 1, 600, 1);
                 //init = 1;
             }
             else   predictiveSettings1.createPredictiveSettings(outputFilenameTrain1, outputFilenameValidation1, numAttributes, 1, 600, 0);

           //  predictiveSettings.setDataFilePath(appset.outFolderPath+"\\JinputInitial.arff");
     }
     else{
             predictiveSettings1.setPath(appset.outFolderPath+"/predictive.s");
             if(init == 0){
                 predictiveSettings1.createPredictiveSettings(outputFilenameTrain1, outputFilenameValidation1, numAttributes,1 , 600, 1);
               //  init = 1;
             }
             else   predictiveSettings1.createPredictiveSettings(outputFilenameTrain1, outputFilenameValidation1, numAttributes, 1, 600, 0);

     }
     
     //call CLUS
     
      exec.run(appset.javaPath,appset.clusPath ,appset.outFolderPath,"predictive.s",0, appset.clusteringMemory);
     
     //load feature ranking
      if(init == 0){
          try{
              
              Path p = null;
              if(appset.system == "windows")
                p= Paths.get(appset.outFolderPath+"\\predictiveTrees600RForest.fimp");
              else  p= Paths.get(appset.outFolderPath+"/predictiveTrees600RForest.fimp");
              BufferedReader read = Files.newBufferedReader(p);
              String line ="";
              int c=0, rank=1;
              while((line = read.readLine())!=null){
                 // System.out.println("line: "+line);
                  if(c<5){
                      c++;
                      continue;
                  }
                  else{
                     String tmpA[] = line.split("\t");
                     String at = tmpA[1].trim();
                     featureRankList1.put(rank++, at);
                     //System.out.println("fr: "+at+" "+rank);
                     featInvRank1.put(at, rank-1);
                  }
              }
              read.close();
              
          }
          catch(IOException e){
              e.printStackTrace();
          }
          init = 1;
      }
      
          try{
              Path p= null;
              BufferedReader read = null;
              String line = "";
              
               if(appset.system == "windows")
                p= Paths.get(appset.outFolderPath+"\\predictive.out");
              else  p= Paths.get(appset.outFolderPath+"/predictive.out");
              
              read = Files.newBufferedReader(p);
              line ="";
              
              int perfLine = 0;
              double auc=0, auprc=0;
              
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
                          auc = Double.parseDouble(tmpT[1].trim());//number format exception, pending!
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
              if(!evalM1.containsKey(numAttributes))
                    evalM1.put(numAttributes, 0.0);
              
              evalM1.put(numAttributes, evalM1.get(numAttributes)+((auprc+auc)/2));
          }
          catch(IOException e){
              e.printStackTrace();
          }
      
      
      //load performance

     //save performance
     
            numAttributes/=2;//cut the list in half in very iteration
            
            for(int z=dataTrain.numAttributes()-2;z>=1;z--){
                if(!featInvRank1.containsKey(dataTrain.attribute(z).name())){
                    System.out.println("Not contained: "+dataTrain.attribute(z).name());
                }
                int attrInd = featInvRank1.get(dataTrain.attribute(z).name());//attributeIndex.get(dataTrain.attribute(z).name());
                if(attrInd>=numAttributes){
                    dataTrain.deleteAttributeAt(z);
                    dataValidation.deleteAttributeAt(z);
                            }
            }
            
            
                      if(appset.system.equals("windows"))
                                outputFilenameValidation1 = appset.outFolderPath+"\\ValidationAA.arff";
                     else  outputFilenameValidation1 = appset.outFolderPath+"/ValidationAA.arff";//save all attrs to file
                     DataSink.write(outputFilenameValidation1, dataValidation);
                     
                      if(appset.system.equals("windows"))
                             outputFilenameTrain1 = appset.outFolderPath+"\\TrainAA.arff";
                      else  outputFilenameTrain1 = appset.outFolderPath+"/TrainAA.arff";
                     DataSink.write(outputFilenameTrain1, dataTrain);
            
            //reduce features
             //use attribute index + feature Ranking to reduce features
 
      }
   }
     }
     catch(Exception e){
         e.printStackTrace();
     }
     
     
     //get the required number of features
     Iterator<Integer> it2 = evalM1.keySet().iterator();
     int numFeatSel = -1;
     double maxS = -1;
     
     while(it2.hasNext()){
         int i = it2.next();
         double sc = evalM1.get(i);
         System.out.println("FL: "+i+" "+sc);
         if(sc>maxS){
             numFeatSel = i;
             maxS = sc;
         }
     }
     
     
     if(appset.system.equals("windows")){
                dataTrainValidation = wdl.loadDataset(appset.outFolderPath+"\\FeatureConstructionTrainValidationRules.arff");
                dataTest = wdl.loadDataset(appset.outFolderPath+"\\FeatureConstructionTestRules.arff");
                dataTest.setClassIndex(dataTrainValidation.numAttributes()-1);
                dataTrainValidation.setClassIndex(dataTrainValidation.numAttributes()-1);
           }
           else{
                dataTrainValidation = wdl.loadDataset(appset.outFolderPath+"/FeatureConstructionTrainValidationRules.arff");
                dataTest = wdl.loadDataset(appset.outFolderPath+"/FeatureConstructionTestRules.arff");
                dataTest.setClassIndex(dataTest.numAttributes()-1);
                dataTrainValidation.setClassIndex(dataTest.numAttributes()-1);
           }
     
     predictiveSettings1 = new SettingsReader();
     int numAttributes = dataTrainValidation.numAttributes();
     String out = appset.outFolderPath;
       if(appset.system.equals("windows")){//createPredictiveSettingsFunction
           out+="\\FeatureConstructionTrainValidationRules.arff";
             predictiveSettings1.setPath(appset.outFolderPath+"\\predictiveFr.s");//replace view1 with real name

                 predictiveSettings1.createPredictiveSettings(out, out, numAttributes, 1, 600, 1);
                 //init = 1;
           //  predictiveSettings.setDataFilePath(appset.outFolderPath+"\\JinputInitial.arff");
     }
     else{
           out+="/FeatureConstructionTrainValidationRules.arff";
             predictiveSettings1.setPath(appset.outFolderPath+"/predictiveFr.s");

                 predictiveSettings1.createPredictiveSettings(out, out, numAttributes,1 , 600, 1);
     }
     
     //call CLUS
     
      exec.run(appset.javaPath,appset.clusPath ,appset.outFolderPath,"predictiveFr.s",0, appset.clusteringMemory);
     
     //load feature ranking
          try{
              featureRankList1.clear();
              featInvRank1.clear();
              
              Path p = null;
              if(appset.system == "windows")
                p= Paths.get(appset.outFolderPath+"\\predictiveFrTrees600RForest.fimp");
              else  p= Paths.get(appset.outFolderPath+"/predictiveFrTrees600RForest.fimp");
              BufferedReader read = Files.newBufferedReader(p);
              String line ="";
              int c=0, rank=1;
              while((line = read.readLine())!=null){
                 // System.out.println("line: "+line);
                  if(c<5){
                      c++;
                      continue;
                  }
                  else{
                     String tmpA[] = line.split("\t");
                     String at = tmpA[1].trim();
                     featureRankList1.put(rank++, at);
                     //System.out.println("fr: "+at+" "+rank);
                     featInvRank1.put(at, rank-1);
                  }
              }
              read.close();
              
          }
          catch(IOException e){
              e.printStackTrace();
          }
      
      if(numFeatSel == -1)
          numFeatSel = dataTrainValidation.numAttributes();
     System.out.println("Feature statistics: "+numFeatSel+" "+maxS);
     //perform feature ranking again on TrainValidation set with the given number of features...
     for(int z=dataTrainValidation.numAttributes()-2;z>=1;z--){//remove the required features
                int attrInd = featInvRank1.get(dataTrainValidation.attribute(z).name()); //attributeIndex.get(dataTrainValidation.attribute(z).name());
                if(attrInd>=numFeatSel){
                    dataTrainValidation.deleteAttributeAt(z);
                    dataTest.deleteAttributeAt(z);
                            }
            }
     
     //create final test
     
     //write dataTrainValidation into file
     
     String outputFilenameTrainValidation = "";
     String outputFilenameTest="";
     try{
      if(appset.system.equals("windows"))
                             outputFilenameTrainValidation = appset.outFolderPath+"\\FeatureTrainFinRules.arff";
                      else  outputFilenameTrainValidation = appset.outFolderPath+"/FeatureTrainFinRules.arff";
                     DataSink.write(outputFilenameTrainValidation, dataTrainValidation);
                     
                    
                     if(appset.system.equals("windows"))
                             outputFilenameTest = appset.outFolderPath+"\\FeatureTestFinRules.arff";
                      else  outputFilenameTest = appset.outFolderPath+"/FeatureTestFinRules.arff";
                     DataSink.write(outputFilenameTest, dataTest);
                     
                     
     }catch(Exception e){
         e.printStackTrace();
     }
     
     //write test into file
     
     //perform train+validation training /test evaluation
     
     predictiveSettings1 = new SettingsReader();
     numAttributes = dataTrainValidation.numAttributes();
     
      if(appset.system.equals("windows")){//createPredictiveSettingsFunction
             predictiveSettings1.setPath(appset.outFolderPath+"\\predictiveFinFCR.s");//replace view1 with real name
            predictiveSettings1.createPredictiveSettings(outputFilenameTrainValidation, outputFilenameTest, numAttributes, 1, 600, 0);

           //  predictiveSettings.setDataFilePath(appset.outFolderPath+"\\JinputInitial.arff");
     }
     else{
             predictiveSettings1.setPath(appset.outFolderPath+"/predictiveFinFCR.s");
               predictiveSettings1.createPredictiveSettings(outputFilenameTrainValidation, outputFilenameTest, numAttributes, 1, 600, 0);

     }
     
     //call CLUS
      
      
     
      exec.run(appset.javaPath,appset.clusPath ,appset.outFolderPath,"predictiveFinFCR.s",0, appset.clusteringMemory);
     
      //call weka models for FC features
       clsf = new J48();
      dataTrainValidation.deleteAttributeAt(0);
      dataTest.deleteAttributeAt(0);
     try{
            clsf.buildClassifier(dataTrainValidation);
            Evaluation eval = new Evaluation(dataTrainValidation);
            eval.evaluateModel(clsf, dataTest); 
            double auprc = 0, auc = 0;
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
            System.out.println(eval.toSummaryString("\nResults FCR\n=====\n",false));
            
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
            System.out.println(eval.toSummaryString("\nResults FCR\n=====\n",false));
            
             if(!resultsWeka.containsKey("NB")){
                resultsWeka.put("NB", new ArrayList<>());
            }
            
            resultsWeka.get("NB").add(auprc);
             resultsWeka.get("NB").add(auc);
            
            clsf = new Logistic();
            
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
            System.out.println("Av.AUPRC Log: "+auprc);
            System.out.println("Av.AUC Log: "+auc);
            System.out.println(eval.toSummaryString("\nResults FCR\n=====\n",false));
            
            
             if(!resultsWeka.containsKey("Log")){
                resultsWeka.put("Log", new ArrayList<>());
            }
            
            resultsWeka.get("Log").add(auprc);
             resultsWeka.get("Log").add(auc);
            
            clsf = new MultilayerPerceptron();
            
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
            System.out.println("Av.AUPRC MLP: "+auprc);
            System.out.println("Av.AUC MLP: "+auc);
            System.out.println(eval.toSummaryString("\nResults FCR\n=====\n",false));
            
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
            System.out.println(eval.toSummaryString("\nResults FCR\n=====\n",false));
            
            
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
            System.out.println(eval.toSummaryString("\nResults FCR\n=====\n",false));
            
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
            System.out.println(eval.toSummaryString("\nResults FCR\n=====\n",false));
            
             if(!resultsWeka.containsKey("LMT")){
                resultsWeka.put("LMT", new ArrayList<>());
            }
            
            resultsWeka.get("LMT").add(auprc);
             resultsWeka.get("LMT").add(auc);
      }
      catch(Exception e){
          e.printStackTrace();
      }
      
      //end of orig features + rule features
      System.out.println("Original features + rules computed!");
     
     if(appset.system.equals("windows")){
            dataTrainValidation = wdl.loadDataset(appset.outFolderPath+"\\TrainValidation.arff");
            dataTest=wdl.loadDataset(appset.outFolderPath+"\\Test.arff");
     }
     else{
        dataTrainValidation = wdl.loadDataset(appset.outFolderPath+"/TrainValidation.arff");
            dataTest=wdl.loadDataset(appset.outFolderPath+"/Test.arff"); 
     }
     
      dataTest.setClassIndex(dataTest.numAttributes()-1);
      dataTrainValidation.setClassIndex(dataTrainValidation.numAttributes()-1);
     
      data = wdl.loadDataset(appset.viewInputPaths.get(0));//load train/validation/test, than add to all
      data.setClassIndex(data.numAttributes()-1);  
      
      ArrayList<String> entityNames=new ArrayList<>();
      
      for(int i=0;i<data.numInstances();i++){
          entityNames.add(data.instance(i).stringValue(0));
         // System.out.println(entityNames.get(i));
      }
      
      int indO = data.numAttributes()-1;
      
      System.out.println("Num attrs old: "+indO);
      
      for(int i=0;i<rrAll.rules.size();i++){
          data.insertAttributeAt(new Attribute("r"+i), data.numAttributes()-1);
          dataTest.insertAttributeAt(new Attribute("r"+i), dataTest.numAttributes()-1);
          dataTrainValidation.insertAttributeAt(new Attribute("r"+i), dataTrainValidation.numAttributes()-1);
      }

      for(int i=0;i<predFinal.redescriptions.size();i++){
           data.insertAttributeAt(new Attribute("R"+i), data.numAttributes()-1);
          dataTest.insertAttributeAt(new Attribute("R"+i), dataTest.numAttributes()-1);
          dataTrainValidation.insertAttributeAt(new Attribute("R"+i), dataTrainValidation.numAttributes()-1);
      }
      
      System.out.println("Num attrs new: "+data.numAttributes());
      //datJFull
      String entN="";
     
      int totNum = indO;
      
     for(int j=0;j<rrAll.rules.size();j++) 
      for(int i=0;i<data.numInstances();i++){
          entN = data.instance(i).stringValue(0);
          if(rrAll.rules.get(j).elements == null){
              System.out.println("Rule with null elements: "+j+"\n "+rrAll.rules.get(j).rule);
          }
          if(!fidFull.exampleId.containsKey(entN)){
              System.out.println("Not contained: "+entN);
              if(fidFull.exampleId.containsKey("\"+entN+\""))
                  entN = "\""+entN+"\"";
          }
          if(rrAll.rules.get(j).elements.contains(fidFull.exampleId.get(entN)))
                    data.instance(i).setValue(indO+j, 1);
          else  data.instance(i).setValue(indO+j, 0);
      }
     
     //napraviti funkciju koja doda sve elemente u element skup redescriptiona
     ArrayList<TIntHashSet> entAll = new ArrayList<>();
     ArrayList<TIntHashSet> tmp = new ArrayList<>();
     
     for(int j=0;j<predFinal.redescriptions.size();j++){
         predFinal.redescriptions.get(j).closeInterval(datJ, fid);
         tmp = predFinal.redescriptions.get(j).computeElementsGen1(datJ, fid,datJFull,fidFull);
         
         entAll.add(new TIntHashSet(datJFull.numExamples));
         TIntHashSet f = tmp.get(0);
         TIntIterator it = f.iterator();
         while(it.hasNext()){
             int el = it.next();
             int cont = 1;
             for(int i=1;i<tmp.size();i++){
                    if(!tmp.get(i).contains(el)){
                        cont = 0;
                        break;
                    }
            }
             if(cont == 1)
                 entAll.get(j).add(el);
         }
         System.out.println("supp R_j: "+entAll.get(j).size());
         
     }
     
     for(int j=0;j<predFinal.redescriptions.size();j++){
          for(int i=0;i<data.numInstances();i++){
          entN = data.instance(i).stringValue(0);
          if(entAll.get(j).contains(fidFull.exampleId.get(entN)))
                    data.instance(i).setValue(indO+j+rrAll.rules.size(), 1);
          else  data.instance(i).setValue(indO+j+rrAll.rules.size(), 0);
      }
     }
      
     String predFile = "";
     try{
      if(appset.system.equals("windows"))//write new file to disk
                        predFile= appset.outFolderPath+"\\FeatureConstruction.arff";
                    else predFile= appset.outFolderPath+"/FeatureConstruction.arff";
                     DataSink.write(predFile, data);
     }
     catch(Exception e){
         e.printStackTrace();
     } //all data
     
      for(int j=0;j<rrAll.rules.size();j++) 
      for(int i=0;i<dataTrainValidation.numInstances();i++){
          entN = dataTrainValidation.instance(i).stringValue(0);
          if(rrAll.rules.get(j).elements.contains(fidFull.exampleId.get(entN)))
                    dataTrainValidation.instance(i).setValue(indO+j, 1);
          else  dataTrainValidation.instance(i).setValue(indO+j, 0);
      }
      
     
     for(int j=0;j<predFinal.redescriptions.size();j++){
          for(int i=0;i<dataTrainValidation.numInstances();i++){
          entN = dataTrainValidation.instance(i).stringValue(0);
          if(entAll.get(j).contains(fidFull.exampleId.get(entN)))
                    dataTrainValidation.instance(i).setValue(indO+j+rrAll.rules.size(), 1);
          else  dataTrainValidation.instance(i).setValue(indO+j+rrAll.rules.size(), 0);
      }
     }
      
     predFile = "";
     try{
      if(appset.system.equals("windows"))//write new file to disk
                        predFile= appset.outFolderPath+"\\FeatureConstructionTrainValidation.arff";
                    else predFile= appset.outFolderPath+"/FeatureConstructionTrainValidation.arff";
                     DataSink.write(predFile, dataTrainValidation);
     }
     catch(Exception e){
         e.printStackTrace();
     }//train-validation data
     
     
      for(int j=0;j<rrAll.rules.size();j++) 
      for(int i=0;i<dataTest.numInstances();i++){
          entN = dataTest.instance(i).stringValue(0);
          if(rrAll.rules.get(j).elements.contains(fidFull.exampleId.get(entN)))
                    dataTest.instance(i).setValue(indO+j, 1);
          else  dataTest.instance(i).setValue(indO+j, 0);
      }
     
     for(int j=0;j<predFinal.redescriptions.size();j++){
          for(int i=0;i<dataTest.numInstances();i++){
          entN = dataTest.instance(i).stringValue(0);
          if(entAll.get(j).contains(fidFull.exampleId.get(entN)))
                    dataTest.instance(i).setValue(indO+j+rrAll.rules.size(), 1);
          else  dataTest.instance(i).setValue(indO+j+rrAll.rules.size(), 0);
      }
     }
      
     predFile = "";
     try{
      if(appset.system.equals("windows"))//write new file to disk
                        predFile= appset.outFolderPath+"\\FeatureConstructionTest.arff";
                    else predFile= appset.outFolderPath+"/FeatureConstructionTest.arff";
                     DataSink.write(predFile, dataTest); 
     }
     catch(Exception e){
         e.printStackTrace();
     }//test data
     
     //compute evaluation using all features
     SettingsReader predictiveSettings = new SettingsReader();
     
     String predTV="",pTest = "";
     
     if(appset.system.equals("windows")){//write new file to disk
                        predTV= appset.outFolderPath+"\\FeatureConstructionTrainValidation.arff";
                        pTest = appset.outFolderPath+"\\FeatureConstructionTest.arff";
     }
     else{ predTV= appset.outFolderPath+"/FeatureConstructionTrainValidation.arff";
            pTest= appset.outFolderPath+"/FeatureConstructionTest.arff";
     }
     
     
     if(appset.system.equals("windows")){//createPredictiveSettingsFunction
             predictiveSettings.setPath(appset.outFolderPath+"\\predictiveAll.s");//replace view1 with real name
                 predictiveSettings.createPredictiveSettings(predTV, pTest, dataTrainValidation.numAttributes(), 1, 600, 0);

     }
     else{
             predictiveSettings.setPath(appset.outFolderPath+"/predictiveAll.s");
                 predictiveSettings.createPredictiveSettings(predTV, pTest, dataTrainValidation.numAttributes(),1 , 600, 0);
     }
     
     //call CLUS
   //   ClusProcessExecutor exec=new ClusProcessExecutor();
     
      exec.run(appset.javaPath,appset.clusPath ,appset.outFolderPath,"predictiveAll.s",0, appset.clusteringMemory);

     //call clus settings and call CLUS
     //create weka evaluation
      
       dataTrainValidationAll = null;
       dataTestAll = null;
      
       if(appset.system.equals("windows")){
                dataTrainValidationAll = wdl.loadDataset(appset.outFolderPath+"\\FeatureConstructionTrainValidation.arff");
                dataTestAll = wdl.loadDataset(appset.outFolderPath+"\\FeatureConstructionTest.arff");
                dataTestAll.setClassIndex(dataTest.numAttributes()-1);
                dataTrainValidationAll.setClassIndex(dataTrainValidation.numAttributes()-1);
           }
           else{
                dataTrainValidationAll = wdl.loadDataset(appset.outFolderPath+"/FeatureConstructionTrainValidation.arff");
                dataTestAll = wdl.loadDataset(appset.outFolderPath+"/FeatureConstructionTest.arff");
                dataTestAll.setClassIndex(dataTestAll.numAttributes()-1);
                dataTrainValidationAll.setClassIndex(dataTrainValidationAll.numAttributes()-1);
           }
       
       //HashMap<String,ArrayList<Double>> resultsWeka = new HashMap<>();
       /*Classifier*/ clsf = new J48();
      dataTrainValidationAll.deleteAttributeAt(0);
      dataTestAll.deleteAttributeAt(0);
      try{
            clsf.buildClassifier(dataTrainValidationAll);
            Evaluation eval = new Evaluation(dataTrainValidationAll);
            eval.evaluateModel(clsf, dataTestAll); 
            double auprc = 0, auc = 0;
            for(int i=0;i<dataTrainValidationAll.classAttribute().numValues();i++){
                System.out.println("AUPRC"+i+": "+eval.areaUnderPRC(i));
                System.out.println("AUC"+i+": "+eval.areaUnderROC(i));
                auprc+=eval.areaUnderPRC(i); auc+=eval.areaUnderROC(i);
            }
            auprc/=dataTrainValidationAll.classAttribute().numValues();
            auc/=dataTrainValidationAll.classAttribute().numValues();
            /*System.out.println("AUPRC0: "+eval.areaUnderPRC(0));
            System.out.println("AUC0: "+eval.areaUnderROC(0));
            System.out.println("AUPRC1: "+eval.areaUnderPRC(1));
            System.out.println("AUC1: "+eval.areaUnderROC(1));*/
            System.out.println("Av.AUPRC J48: "+auprc);
            System.out.println("Av.AUC J48: "+auc);
            System.out.println(eval.toSummaryString("\nResults FCAll\n=====\n",false));
            
            if(!resultsWeka.containsKey("J48")){
                resultsWeka.put("J48", new ArrayList<>());
            }
            
            resultsWeka.get("J48").add(auprc);
             resultsWeka.get("J48").add(auc);
            
            clsf = new NaiveBayes();
            
             clsf.buildClassifier(dataTrainValidationAll);
            eval = new Evaluation(dataTrainValidationAll);
            eval.evaluateModel(clsf, dataTestAll);
             auprc = 0; auc = 0;
            for(int i=0;i<dataTrainValidationAll.classAttribute().numValues();i++){
                System.out.println("AUPRC"+i+": "+eval.areaUnderPRC(i));
                System.out.println("AUC"+i+": "+eval.areaUnderROC(i));
                auprc+=eval.areaUnderPRC(i); auc+=eval.areaUnderROC(i);
            }
            auprc/=dataTrainValidationAll.classAttribute().numValues();
            auc/=dataTrainValidationAll.classAttribute().numValues();
            System.out.println("Av.AUPRC NB: "+auprc);
            System.out.println("Av.AUC NB: "+auc);
            System.out.println(eval.toSummaryString("\nResults FCAll\n=====\n",false));
            
             if(!resultsWeka.containsKey("NB")){
                resultsWeka.put("NB", new ArrayList<>());
            }
            
            resultsWeka.get("NB").add(auprc);
             resultsWeka.get("NB").add(auc);
            
            clsf = new Logistic();
            
             clsf.buildClassifier(dataTrainValidationAll);
            eval = new Evaluation(dataTrainValidationAll);
            eval.evaluateModel(clsf, dataTestAll);
             auprc = 0; auc = 0;
            for(int i=0;i<dataTrainValidationAll.classAttribute().numValues();i++){
                System.out.println("AUPRC"+i+": "+eval.areaUnderPRC(i));
                System.out.println("AUC"+i+": "+eval.areaUnderROC(i));
                auprc+=eval.areaUnderPRC(i); auc+=eval.areaUnderROC(i);
            }
            auprc/=dataTrainValidationAll.classAttribute().numValues();
            auc/=dataTrainValidationAll.classAttribute().numValues();
            System.out.println("Av.AUPRC Log: "+auprc);
            System.out.println("Av.AUC Log: "+auc);
            System.out.println(eval.toSummaryString("\nResults FCAll\n=====\n",false));
            
            
             if(!resultsWeka.containsKey("Log")){
                resultsWeka.put("Log", new ArrayList<>());
            }
            
            resultsWeka.get("Log").add(auprc);
             resultsWeka.get("Log").add(auc);
            
            clsf = new MultilayerPerceptron();
            
             clsf.buildClassifier(dataTrainValidationAll);
            eval = new Evaluation(dataTrainValidationAll);
            eval.evaluateModel(clsf, dataTestAll);
             auprc = 0; auc = 0;
            for(int i=0;i<dataTrainValidationAll.classAttribute().numValues();i++){
                System.out.println("AUPRC"+i+": "+eval.areaUnderPRC(i));
                System.out.println("AUC"+i+": "+eval.areaUnderROC(i));
                auprc+=eval.areaUnderPRC(i); auc+=eval.areaUnderROC(i);
            }
            auprc/=dataTrainValidationAll.classAttribute().numValues();
            auc/=dataTrainValidationAll.classAttribute().numValues();
            System.out.println("Av.AUPRC MLP: "+auprc);
            System.out.println("Av.AUC MLP: "+auc);
            System.out.println(eval.toSummaryString("\nResults FCAll\n=====\n",false));
            
             if(!resultsWeka.containsKey("MLP")){
                resultsWeka.put("MLP", new ArrayList<>());
            }
            
            resultsWeka.get("MLP").add(auprc);
             resultsWeka.get("MLP").add(auc);
            
            clsf = new KStar(); 
            
             clsf.buildClassifier(dataTrainValidationAll);
            eval = new Evaluation(dataTrainValidationAll);
            eval.evaluateModel(clsf, dataTestAll);
             auprc = 0; auc = 0;
            for(int i=0;i<dataTrainValidationAll.classAttribute().numValues();i++){
                System.out.println("AUPRC"+i+": "+eval.areaUnderPRC(i));
                System.out.println("AUC"+i+": "+eval.areaUnderROC(i));
                auprc+=eval.areaUnderPRC(i); auc+=eval.areaUnderROC(i);
            }
            auprc/=dataTrainValidationAll.classAttribute().numValues();
            auc/=dataTrainValidationAll.classAttribute().numValues();
            System.out.println("Av.AUPRC KStar: "+auprc);
            System.out.println("Av.AUC KStar: "+auc);
            System.out.println(eval.toSummaryString("\nResults FCAll\n=====\n",false));
            
            
             if(!resultsWeka.containsKey("KS")){
                resultsWeka.put("KS", new ArrayList<>());
            }
            
            resultsWeka.get("KS").add(auprc);
             resultsWeka.get("KS").add(auc);
            
            clsf = new DecisionStump();
            
             clsf.buildClassifier(dataTrainValidationAll);
            eval = new Evaluation(dataTrainValidationAll);
            eval.evaluateModel(clsf, dataTestAll);
             auprc = 0; auc = 0;
            for(int i=0;i<dataTrainValidationAll.classAttribute().numValues();i++){
                System.out.println("AUPRC"+i+": "+eval.areaUnderPRC(i));
                System.out.println("AUC"+i+": "+eval.areaUnderROC(i));
                auprc+=eval.areaUnderPRC(i); auc+=eval.areaUnderROC(i);
            }
            auprc/=dataTrainValidationAll.classAttribute().numValues();
            auc/=dataTrainValidationAll.classAttribute().numValues();
            System.out.println("Av.AUPRC DStump: "+auprc);
            System.out.println("Av.AUC DStump: "+auc);
            System.out.println(eval.toSummaryString("\nResults FCAll\n=====\n",false));
            
             if(!resultsWeka.containsKey("DSt")){
                resultsWeka.put("DSt", new ArrayList<>());
            }
            
            resultsWeka.get("DSt").add(auprc);
             resultsWeka.get("DSt").add(auc);
            
            clsf = new LMT();
            
             clsf.buildClassifier(dataTrainValidationAll);
            eval = new Evaluation(dataTrainValidationAll);
            eval.evaluateModel(clsf, dataTestAll);
             auprc = 0; auc = 0;
            for(int i=0;i<dataTrainValidationAll.classAttribute().numValues();i++){
                System.out.println("AUPRC"+i+": "+eval.areaUnderPRC(i));
                System.out.println("AUC"+i+": "+eval.areaUnderROC(i));
                auprc+=eval.areaUnderPRC(i); auc+=eval.areaUnderROC(i);
            }
            auprc/=dataTrainValidationAll.classAttribute().numValues();
            auc/=dataTrainValidationAll.classAttribute().numValues();
            System.out.println("Av.AUPRC LMT: "+auprc);
            System.out.println("Av.AUC LMT: "+auc);
            System.out.println(eval.toSummaryString("\nResults FCAll\n=====\n",false));
            
             if(!resultsWeka.containsKey("LMT")){
                resultsWeka.put("LMT", new ArrayList<>());
            }
            
            resultsWeka.get("LMT").add(auprc);
             resultsWeka.get("LMT").add(auc);
      }
      catch(Exception e){
          e.printStackTrace();
      }
     
     HashMap<Integer,Double> evalM = new HashMap<>();//hashmap containing numFeatures + average score = (convex combination of AUC and AUPRC) 
     TreeMap<Integer,String> featureRankList = new TreeMap<>();//rank list sorted by feature importance (first - best, last-worst)
     HashMap<String,Integer> featInvRank = new HashMap<>();
     HashMap<String,Integer> attributeIndex = new HashMap<>();//map containing index of each attribute in the dataset
       HashMap<Integer,String> indexAttribute = new HashMap<>();
     String outputFilenameTrain = "", outputFilenameValidation="";
       exec=new ClusProcessExecutor();
     try{
    for(int numTries = 0; numTries<20; numTries++){ //20 random splits to train/validation
        System.out.println("Feature selection iteration: "+(numTries+1)); 
        featureRankList.clear(); featInvRank.clear();
         dataTrainValidation.randomize(new java.util.Random(numTries));//create different split in each iteration
         
         insfilt = new InstancesFilter(dataTrainValidation);
                     insfilt.removeStratifiedFoldsFilter(1,4,true);
                     dataValidation = insfilt.getFilteredInstances();
                     
                     insfilt = new InstancesFilter(dataTrainValidation);
                     insfilt.removeStratifiedFoldsFilter(1,4,false);
                     dataTrain = insfilt.getFilteredInstances();
                     
                      if(appset.system.equals("windows"))
                                outputFilenameValidation = appset.outFolderPath+"\\ValidationAA.arff";
                     else  outputFilenameValidation = appset.outFolderPath+"/ValidationAA.arff";//save all attrs to file
                     DataSink.write(outputFilenameValidation, dataValidation);
                     
                      if(appset.system.equals("windows"))
                             outputFilenameTrain = appset.outFolderPath+"\\TrainAA.arff";
                      else  outputFilenameTrain = appset.outFolderPath+"/TrainAA.arff";
                     DataSink.write(outputFilenameTrain, dataTrain);
        
         numAttributes = dataTrainValidation.numAttributes();
        //split to train and validation
        //save to disk
     int init=0;
        while(numAttributes>10){
            System.out.println("numAttributes: "+numAttributes);
            System.out.println("naTrain: "+dataTrain.numAttributes());
     //create predictive settings
      predictiveSettings = new SettingsReader();
     numAttributes = dataTrain.numAttributes();
     //load attribte index
     attributeIndex.clear();
     indexAttribute.clear();
     
     for(int z=0;z<dataTrain.numAttributes();z++){
         indexAttribute.put(z, dataTrain.attribute(z).name());
         attributeIndex.put(dataTrain.attribute(z).name(), z);
     }
     
     
     if(appset.system.equals("windows")){//createPredictiveSettingsFunction
             predictiveSettings.setPath(appset.outFolderPath+"\\predictive.s");//replace view1 with real name
             if(init == 0){
                 predictiveSettings.createPredictiveSettings(outputFilenameTrain, outputFilenameValidation, numAttributes, 1, 600, 1);
                 //init = 1;
             }
             else   predictiveSettings.createPredictiveSettings(outputFilenameTrain, outputFilenameValidation, numAttributes, 1, 600, 0);

           //  predictiveSettings.setDataFilePath(appset.outFolderPath+"\\JinputInitial.arff");
     }
     else{
             predictiveSettings.setPath(appset.outFolderPath+"/predictive.s");
             if(init == 0){
                 predictiveSettings.createPredictiveSettings(outputFilenameTrain, outputFilenameValidation, numAttributes,1 , 600, 1);
               //  init = 1;
             }
             else   predictiveSettings.createPredictiveSettings(outputFilenameTrain, outputFilenameValidation, numAttributes, 1, 600, 0);

     }
     
     //call CLUS
     
      exec.run(appset.javaPath,appset.clusPath ,appset.outFolderPath,"predictive.s",0, appset.clusteringMemory);
     
     //load feature ranking
      if(init == 0){
          try{
              
              Path p = null;
              if(appset.system == "windows")
                p= Paths.get(appset.outFolderPath+"\\predictiveTrees600RForest.fimp");
              else  p= Paths.get(appset.outFolderPath+"/predictiveTrees600RForest.fimp");
              BufferedReader read = Files.newBufferedReader(p);
              String line ="";
              int c=0, rank=1;
              while((line = read.readLine())!=null){
                 // System.out.println("line: "+line);
                  if(c<5){
                      c++;
                      continue;
                  }
                  else{
                     String tmpA[] = line.split("\t");
                     String at = tmpA[1].trim();
                     featureRankList.put(rank++, at);
                     //System.out.println("fr: "+at+" "+rank);
                     featInvRank.put(at, rank-1);
                  }
              }
              read.close();
              
          }
          catch(IOException e){
              e.printStackTrace();
          }
          init = 1;
      }
      
          try{
              Path p= null;
              BufferedReader read = null;
              String line = "";
              
               if(appset.system == "windows")
                p= Paths.get(appset.outFolderPath+"\\predictive.out");
              else  p= Paths.get(appset.outFolderPath+"/predictive.out");
              
              read = Files.newBufferedReader(p);
              line ="";
              
              int perfLine = 0;
              double auc=0, auprc=0;
              
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
                          auc = Double.parseDouble(tmpT[1].trim());//number format exception, pending!
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
              if(!evalM.containsKey(numAttributes))
                    evalM.put(numAttributes, 0.0);
              
              evalM.put(numAttributes, evalM.get(numAttributes)+((auprc+auc)/2));
          }
          catch(IOException e){
              e.printStackTrace();
          }
      
      
      //load performance

     //save performance
     
            numAttributes/=2;//cut the list in half in very iteration
            
            for(int z=dataTrain.numAttributes()-2;z>=1;z--){
                if(!featInvRank.containsKey(dataTrain.attribute(z).name())){
                    System.out.println("Not contained: "+dataTrain.attribute(z).name());
                }
                int attrInd = featInvRank.get(dataTrain.attribute(z).name());//attributeIndex.get(dataTrain.attribute(z).name());
                if(attrInd>=numAttributes){
                    dataTrain.deleteAttributeAt(z);
                    dataValidation.deleteAttributeAt(z);
                            }
            }
            
            
                      if(appset.system.equals("windows"))
                                outputFilenameValidation = appset.outFolderPath+"\\ValidationAA.arff";
                     else  outputFilenameValidation = appset.outFolderPath+"/ValidationAA.arff";//save all attrs to file
                     DataSink.write(outputFilenameValidation, dataValidation);
                     
                      if(appset.system.equals("windows"))
                             outputFilenameTrain = appset.outFolderPath+"\\TrainAA.arff";
                      else  outputFilenameTrain = appset.outFolderPath+"/TrainAA.arff";
                     DataSink.write(outputFilenameTrain, dataTrain);
            
            //reduce features
             //use attribute index + feature Ranking to reduce features
 
      }
   }
     }
     catch(Exception e){
         e.printStackTrace();
     }
     
     
     //get the required number of features
     Iterator<Integer> it = evalM.keySet().iterator();
      numFeatSel = -1;
      maxS = -1;
     
     while(it.hasNext()){
         int i = it.next();
         double sc = evalM.get(i);
         System.out.println("FL: "+i+" "+sc);
         if(sc>maxS){
             numFeatSel = i;
             maxS = sc;
         }
     }
     
     
     if(appset.system.equals("windows")){
                dataTrainValidation = wdl.loadDataset(appset.outFolderPath+"\\FeatureConstructionTrainValidation.arff");
                dataTest = wdl.loadDataset(appset.outFolderPath+"\\FeatureConstructionTest.arff");
                dataTest.setClassIndex(dataTrainValidation.numAttributes()-1);
                dataTrainValidation.setClassIndex(dataTrainValidation.numAttributes()-1);
           }
           else{
                dataTrainValidation = wdl.loadDataset(appset.outFolderPath+"/FeatureConstructionTrainValidation.arff");
                dataTest = wdl.loadDataset(appset.outFolderPath+"/FeatureConstructionTest.arff");
                dataTest.setClassIndex(dataTest.numAttributes()-1);
                dataTrainValidation.setClassIndex(dataTest.numAttributes()-1);
           }
     
     predictiveSettings = new SettingsReader();
     numAttributes = dataTrainValidation.numAttributes();
     out = appset.outFolderPath;
       if(appset.system.equals("windows")){//createPredictiveSettingsFunction
           out+="\\FeatureConstructionTrainValidation.arff";
             predictiveSettings.setPath(appset.outFolderPath+"\\predictiveF.s");//replace view1 with real name

                 predictiveSettings.createPredictiveSettings(out, out, numAttributes, 1, 600, 1);
                 //init = 1;
           //  predictiveSettings.setDataFilePath(appset.outFolderPath+"\\JinputInitial.arff");
     }
     else{
           out+="/FeatureConstructionTrainValidation.arff";
             predictiveSettings.setPath(appset.outFolderPath+"/predictiveF.s");

                 predictiveSettings.createPredictiveSettings(out, out, numAttributes,1 , 600, 1);
     }
     
     //call CLUS
     
      exec.run(appset.javaPath,appset.clusPath ,appset.outFolderPath,"predictiveF.s",0, appset.clusteringMemory);
     
     //load feature ranking
          try{
              featureRankList.clear();
              featInvRank.clear();
              
              Path p = null;
              if(appset.system == "windows")
                p= Paths.get(appset.outFolderPath+"\\predictiveFTrees600RForest.fimp");
              else  p= Paths.get(appset.outFolderPath+"/predictiveFTrees600RForest.fimp");
              BufferedReader read = Files.newBufferedReader(p);
              String line ="";
              int c=0, rank=1;
              while((line = read.readLine())!=null){
                 // System.out.println("line: "+line);
                  if(c<5){
                      c++;
                      continue;
                  }
                  else{
                     String tmpA[] = line.split("\t");
                     String at = tmpA[1].trim();
                     featureRankList.put(rank++, at);
                     //System.out.println("fr: "+at+" "+rank);
                     featInvRank.put(at, rank-1);
                  }
              }
              read.close();
              
          }
          catch(IOException e){
              e.printStackTrace();
          }
      
      if(numFeatSel == -1)
          numFeatSel = dataTrainValidation.numAttributes();
     System.out.println("Feature statistics: "+numFeatSel+" "+maxS);
     //perform feature ranking again on TrainValidation set with the given number of features...
     for(int z=dataTrainValidation.numAttributes()-2;z>=1;z--){//remove the required features
                int attrInd = featInvRank.get(dataTrainValidation.attribute(z).name()); //attributeIndex.get(dataTrainValidation.attribute(z).name());
                if(attrInd>=numFeatSel){
                    dataTrainValidation.deleteAttributeAt(z);
                    dataTest.deleteAttributeAt(z);
                            }
            }
     
     //create final test
     
     //write dataTrainValidation into file
     
      outputFilenameTrainValidation = "";
      outputFilenameTest="";
     try{
      if(appset.system.equals("windows"))
                             outputFilenameTrainValidation = appset.outFolderPath+"\\FeatureTrainFin.arff";
                      else  outputFilenameTrainValidation = appset.outFolderPath+"/FeatureTrainFin.arff";
                     DataSink.write(outputFilenameTrainValidation, dataTrainValidation);
                     
                    
                     if(appset.system.equals("windows"))
                             outputFilenameTest = appset.outFolderPath+"\\FeatureTestFin.arff";
                      else  outputFilenameTest = appset.outFolderPath+"/FeatureTestFin.arff";
                     DataSink.write(outputFilenameTest, dataTest);
                     
                     
     }catch(Exception e){
         e.printStackTrace();
     }
     
     //write test into file
     
     //perform train+validation training /test evaluation
     
     predictiveSettings = new SettingsReader();
     numAttributes = dataTrainValidation.numAttributes();
     
      if(appset.system.equals("windows")){//createPredictiveSettingsFunction
             predictiveSettings.setPath(appset.outFolderPath+"\\predictiveFinFC.s");//replace view1 with real name
            predictiveSettings.createPredictiveSettings(outputFilenameTrainValidation, outputFilenameTest, numAttributes, 1, 600, 0);

           //  predictiveSettings.setDataFilePath(appset.outFolderPath+"\\JinputInitial.arff");
     }
     else{
             predictiveSettings.setPath(appset.outFolderPath+"/predictiveFinFC.s");
               predictiveSettings.createPredictiveSettings(outputFilenameTrainValidation, outputFilenameTest, numAttributes, 1, 600, 0);

     }
     
     //call CLUS
      
      
     
      exec.run(appset.javaPath,appset.clusPath ,appset.outFolderPath,"predictiveFinFC.s",0, appset.clusteringMemory);
     
      //call weka models for FC features
       clsf = new J48();
      dataTrainValidation.deleteAttributeAt(0);
      dataTest.deleteAttributeAt(0);
     try{
            clsf.buildClassifier(dataTrainValidation);
            Evaluation eval = new Evaluation(dataTrainValidation);
            eval.evaluateModel(clsf, dataTest); 
            double auprc = 0, auc = 0;
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
            
            clsf = new Logistic();
            
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
            System.out.println("Av.AUPRC Log: "+auprc);
            System.out.println("Av.AUC Log: "+auc);
            System.out.println(eval.toSummaryString("\nResults FC\n=====\n",false));
            
            
             if(!resultsWeka.containsKey("Log")){
                resultsWeka.put("Log", new ArrayList<>());
            }
            
            resultsWeka.get("Log").add(auprc);
             resultsWeka.get("Log").add(auc);
            
            clsf = new MultilayerPerceptron();
            
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
      //add different classifiers from weka
      
      System.out.println("All features completed!");
      
     //compare performance with original data
     
       if(appset.system.equals("windows")){
                dataTrainValidation = wdl.loadDataset(appset.outFolderPath+"\\TrainValidation.arff");
                dataTest = wdl.loadDataset(appset.outFolderPath+"\\Test.arff");
                dataTest.setClassIndex(dataTrainValidation.numAttributes()-1);
                dataTrainValidation.setClassIndex(dataTrainValidation.numAttributes()-1);
           }
           else{
                dataTrainValidation = wdl.loadDataset(appset.outFolderPath+"/TrainValidation.arff");
                dataTest = wdl.loadDataset(appset.outFolderPath+"/Test.arff");
                dataTest.setClassIndex(dataTrainValidation.numAttributes()-1);
                dataTrainValidation.setClassIndex(dataTrainValidation.numAttributes()-1);
           }
      
       
       if(appset.system.equals("windows")){//createPredictiveSettingsFunction
             predictiveSettings.setPath(appset.outFolderPath+"\\predictiveOrig.s");//replace view1 with real name
              outputFilenameTrainValidation = appset.outFolderPath+"\\TrainValidation.arff";
              outputFilenameTest = appset.outFolderPath+"\\Test.arff";
            predictiveSettings.createPredictiveSettings(outputFilenameTrainValidation, outputFilenameTest, dataTrainValidation.numAttributes(), 1, 600, 0);

           //  predictiveSettings.setDataFilePath(appset.outFolderPath+"\\JinputInitial.arff");
     }
     else{
             predictiveSettings.setPath(appset.outFolderPath+"/predictiveOrig.s");
              outputFilenameTrainValidation = appset.outFolderPath+"/TrainValidation.arff";
              outputFilenameTest = appset.outFolderPath+"/Test.arff";
               predictiveSettings.createPredictiveSettings(outputFilenameTrainValidation, outputFilenameTest,dataTrainValidation.numAttributes(), 1, 600, 0);

     }
     
     //call CLUS
     
      exec.run(appset.javaPath,appset.clusPath ,appset.outFolderPath,"predictiveOrig.s",0, appset.clusteringMemory);
     
       //call weka models for Orig features
       dataTrainValidation.deleteAttributeAt(0);
      dataTest.deleteAttributeAt(0);
          clsf = new J48();
     try{
            clsf.buildClassifier(dataTrainValidation);
            Evaluation eval = new Evaluation(dataTrainValidation);
            eval.evaluateModel(clsf, dataTest); 
            double auprc = 0, auc = 0;
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
            System.out.println(eval.toSummaryString("\nResults Orig\n=====\n",false));
            
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
            System.out.println(eval.toSummaryString("\nResults Orig\n=====\n",false));
            
             if(!resultsWeka.containsKey("NB")){
                resultsWeka.put("NB", new ArrayList<>());
            }
            
            resultsWeka.get("NB").add(auprc);
             resultsWeka.get("NB").add(auc);
            
            clsf = new Logistic();
            
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
            System.out.println("Av.AUPRC Log: "+auprc);
            System.out.println("Av.AUC Log: "+auc);
            System.out.println(eval.toSummaryString("\nResults Orig\n=====\n",false));
            
            
             if(!resultsWeka.containsKey("Log")){
                resultsWeka.put("Log", new ArrayList<>());
            }
            
            resultsWeka.get("Log").add(auprc);
             resultsWeka.get("Log").add(auc);
            
            clsf = new MultilayerPerceptron();
            
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
            System.out.println("Av.AUPRC MLP: "+auprc);
            System.out.println("Av.AUC MLP: "+auc);
            System.out.println(eval.toSummaryString("\nResults Orig\n=====\n",false));
            
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
            System.out.println(eval.toSummaryString("\nResults Orig\n=====\n",false));
            
            
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
            System.out.println(eval.toSummaryString("\nResults Orig\n=====\n",false));
            
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
            System.out.println(eval.toSummaryString("\nResults Orig\n=====\n",false));
            
             if(!resultsWeka.containsKey("LMT")){
                resultsWeka.put("LMT", new ArrayList<>());
            }
            
            resultsWeka.get("LMT").add(auprc);
             resultsWeka.get("LMT").add(auc);
      
             
             //repeat feature selection on original features...

      if(appset.system.equals("windows")){
                dataTrainValidation = wdl.loadDataset(appset.outFolderPath+"\\TrainValidation.arff");
                dataTest = wdl.loadDataset(appset.outFolderPath+"\\Test.arff");
                dataTest.setClassIndex(dataTest.numAttributes()-1);
                dataTrainValidation.setClassIndex(dataTrainValidation.numAttributes()-1);
           }
           else{
                dataTrainValidation = wdl.loadDataset(appset.outFolderPath+"/TrainValidation.arff");
                dataTest = wdl.loadDataset(appset.outFolderPath+"/Test.arff");
                dataTest.setClassIndex(dataTest.numAttributes()-1);
                dataTrainValidation.setClassIndex(dataTrainValidation.numAttributes()-1);
           }
      
      evalM = new HashMap<>();
       try{
    for(int numTries = 0; numTries<20; numTries++){ //20 random splits to train/validation
        System.out.println("Feature selection iteration: "+(numTries+1)); 
        featureRankList.clear(); featInvRank.clear();
         dataTrainValidation.randomize(new java.util.Random(numTries));//create different split in each iteration
         
         insfilt = new InstancesFilter(dataTrainValidation);
                     insfilt.removeStratifiedFoldsFilter(1,4,true);
                     dataValidation = insfilt.getFilteredInstances();
                     
                     insfilt = new InstancesFilter(dataTrainValidation);
                     insfilt.removeStratifiedFoldsFilter(1,4,false);
                     dataTrain = insfilt.getFilteredInstances();
                     
                      if(appset.system.equals("windows"))
                                outputFilenameValidation = appset.outFolderPath+"\\ValidationAA.arff";
                     else  outputFilenameValidation = appset.outFolderPath+"/ValidationAA.arff";//save all attrs to file
                     DataSink.write(outputFilenameValidation, dataValidation);
                     
                      if(appset.system.equals("windows"))
                             outputFilenameTrain = appset.outFolderPath+"\\TrainAA.arff";
                      else  outputFilenameTrain = appset.outFolderPath+"/TrainAA.arff";
                     DataSink.write(outputFilenameTrain, dataTrain);
        
         numAttributes = dataTrainValidation.numAttributes();
        //split to train and validation
        //save to disk
     int init=0;
        while(numAttributes>10){
            System.out.println("numAttributes: "+numAttributes);
            System.out.println("naTrain: "+dataTrain.numAttributes());
     //create predictive settings
      predictiveSettings = new SettingsReader();
     numAttributes = dataTrain.numAttributes();
     //load attribte index
     attributeIndex.clear();
     indexAttribute.clear();
     
     for(int z=0;z<dataTrain.numAttributes();z++){
         indexAttribute.put(z, dataTrain.attribute(z).name());
         attributeIndex.put(dataTrain.attribute(z).name(), z);
     }
     
     
     if(appset.system.equals("windows")){//createPredictiveSettingsFunction
             predictiveSettings.setPath(appset.outFolderPath+"\\predictive.s");//replace view1 with real name
             if(init == 0){
                 predictiveSettings.createPredictiveSettings(outputFilenameTrain, outputFilenameValidation, numAttributes, 1, 600, 1);
                 //init = 1;
             }
             else   predictiveSettings.createPredictiveSettings(outputFilenameTrain, outputFilenameValidation, numAttributes, 1, 600, 0);

           //  predictiveSettings.setDataFilePath(appset.outFolderPath+"\\JinputInitial.arff");
     }
     else{
             predictiveSettings.setPath(appset.outFolderPath+"/predictive.s");
             if(init == 0){
                 predictiveSettings.createPredictiveSettings(outputFilenameTrain, outputFilenameValidation, numAttributes,1 , 600, 1);
               //  init = 1;
             }
             else   predictiveSettings.createPredictiveSettings(outputFilenameTrain, outputFilenameValidation, numAttributes, 1, 600, 0);

     }
     
     //call CLUS
     
      exec.run(appset.javaPath,appset.clusPath ,appset.outFolderPath,"predictive.s",0, appset.clusteringMemory);
     
     //load feature ranking
      if(init == 0){
          try{
              
              Path p = null;
              if(appset.system == "windows")
                p= Paths.get(appset.outFolderPath+"\\predictiveTrees600RForest.fimp");
              else  p= Paths.get(appset.outFolderPath+"/predictiveTrees600RForest.fimp");
              BufferedReader read = Files.newBufferedReader(p);
              String line ="";
              int c=0, rank=1;
              while((line = read.readLine())!=null){
                 // System.out.println("line: "+line);
                  if(c<5){
                      c++;
                      continue;
                  }
                  else{
                     String tmpA[] = line.split("\t");
                     String at = tmpA[1].trim();
                     featureRankList.put(rank++, at);
                     //System.out.println("fr: "+at+" "+rank);
                     featInvRank.put(at, rank-1);
                  }
              }
              read.close();
              
          }
          catch(IOException e){
              e.printStackTrace();
          }
          init = 1;
      }
      
          try{
              Path p= null;
              BufferedReader read = null;
              String line = "";
              
               if(appset.system.equals("windows"))
                p= Paths.get(appset.outFolderPath+"\\predictive.out");
              else  p= Paths.get(appset.outFolderPath+"/predictive.out");
              
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
              if(!evalM.containsKey(numAttributes))
                    evalM.put(numAttributes, 0.0);
              
              evalM.put(numAttributes, evalM.get(numAttributes)+((auprc+auc)/2));
          }
          catch(IOException e){
              e.printStackTrace();
          }
      
      
      //load performance

     //save performance
     
            numAttributes/=2;//cut the list in half in very iteration
            
            for(int z=dataTrain.numAttributes()-2;z>=1;z--){
                if(!featInvRank.containsKey(dataTrain.attribute(z).name())){
                    System.out.println("Not contained: "+dataTrain.attribute(z).name());
                }
                int attrInd = featInvRank.get(dataTrain.attribute(z).name());//attributeIndex.get(dataTrain.attribute(z).name());
                if(attrInd>=numAttributes){
                    dataTrain.deleteAttributeAt(z);
                    dataValidation.deleteAttributeAt(z);
                            }
            }
            
            
                      if(appset.system.equals("windows"))
                                outputFilenameValidation = appset.outFolderPath+"\\ValidationAA.arff";
                     else  outputFilenameValidation = appset.outFolderPath+"/ValidationAA.arff";//save all attrs to file
                     DataSink.write(outputFilenameValidation, dataValidation);
                     
                      if(appset.system.equals("windows"))
                             outputFilenameTrain = appset.outFolderPath+"\\TrainAA.arff";
                      else  outputFilenameTrain = appset.outFolderPath+"/TrainAA.arff";
                     DataSink.write(outputFilenameTrain, dataTrain);
            
            //reduce features
             //use attribute index + feature Ranking to reduce features
 
      }
   }
     }
     catch(Exception e){
         e.printStackTrace();
     }
     
     
     //get the required number of features
      it = evalM.keySet().iterator();
      numFeatSel = -1;
      maxS = -1;
     
     while(it.hasNext()){
         int i = it.next();
         double sc = evalM.get(i);
         System.out.println("FL: "+i+" "+sc);
         if(sc>maxS){
             numFeatSel = i;
             maxS = sc;
         }
     }
     
     
     if(appset.system.equals("windows")){
                dataTrainValidation = wdl.loadDataset(appset.outFolderPath+"\\TrainValidation.arff");
                dataTest = wdl.loadDataset(appset.outFolderPath+"\\Test.arff");
                dataTest.setClassIndex(dataTrainValidation.numAttributes()-1);
                dataTrainValidation.setClassIndex(dataTrainValidation.numAttributes()-1);
           }
           else{
                dataTrainValidation = wdl.loadDataset(appset.outFolderPath+"/TrainValidation.arff");
                dataTest = wdl.loadDataset(appset.outFolderPath+"/Test.arff");
                dataTest.setClassIndex(dataTest.numAttributes()-1);
                dataTrainValidation.setClassIndex(dataTest.numAttributes()-1);
           }
     
     predictiveSettings = new SettingsReader();
      numAttributes = dataTrainValidation.numAttributes();
     out = appset.outFolderPath;
       if(appset.system.equals("windows")){//createPredictiveSettingsFunction
           out+="\\TrainValidation.arff";
             predictiveSettings.setPath(appset.outFolderPath+"\\predictiveOrigF.s");//replace view1 with real name

                 predictiveSettings.createPredictiveSettings(out, out, numAttributes, 1, 600, 1);
     }
     else{
           out+="/TrainValidation.arff";
             predictiveSettings.setPath(appset.outFolderPath+"/predictiveOrigF.s");

                 predictiveSettings.createPredictiveSettings(out, out, numAttributes,1 , 600, 1);
     }
     
     //call CLUS
     
      exec.run(appset.javaPath,appset.clusPath ,appset.outFolderPath,"predictiveOrigF.s",0, appset.clusteringMemory);
     
     //load feature ranking
          try{
              featureRankList.clear();
              featInvRank.clear();
              
              Path p = null;
              if(appset.system == "windows")
                p= Paths.get(appset.outFolderPath+"\\predictiveOrigFTrees600RForest.fimp");
              else  p= Paths.get(appset.outFolderPath+"/predictiveOrigFTrees600RForest.fimp");
              BufferedReader read = Files.newBufferedReader(p);
              String line ="";
              int c=0, rank=1;
              while((line = read.readLine())!=null){
                 // System.out.println("line: "+line);
                  if(c<5){
                      c++;
                      continue;
                  }
                  else{
                     String tmpA[] = line.split("\t");
                     String at = tmpA[1].trim();
                     featureRankList.put(rank++, at);
                     //System.out.println("fr: "+at+" "+rank);
                     featInvRank.put(at, rank-1);
                  }
              }
              read.close();
              
          }
          catch(IOException e){
              e.printStackTrace();
          }
      
       if(numFeatSel == -1)
          numFeatSel = dataTrainValidation.numAttributes();
     System.out.println("Feature statistics: "+numFeatSel+" "+maxS);
     //perform feature ranking again on TrainValidation set with the given number of features...
     for(int z=dataTrainValidation.numAttributes()-2;z>=1;z--){//remove the required features
                int attrInd = featInvRank.get(dataTrainValidation.attribute(z).name()); //attributeIndex.get(dataTrainValidation.attribute(z).name());
                if(attrInd>=numFeatSel){
                    dataTrainValidation.deleteAttributeAt(z);
                    dataTest.deleteAttributeAt(z);
                            }
            }
     
     //create final test
     
     //write dataTrainValidation into file
     
      outputFilenameTrainValidation = "";
     outputFilenameTest="";
     try{
      if(appset.system.equals("windows"))
                             outputFilenameTrainValidation = appset.outFolderPath+"\\FeatureOrigTrainFin.arff";
                      else  outputFilenameTrainValidation = appset.outFolderPath+"/FeatureOrigTrainFin.arff";
                     DataSink.write(outputFilenameTrainValidation, dataTrainValidation);
                     
                    
                     if(appset.system.equals("windows"))
                             outputFilenameTest = appset.outFolderPath+"\\FeatureOrigTestFin.arff";
                      else  outputFilenameTest = appset.outFolderPath+"/FeatureOrigTestFin.arff";
                     DataSink.write(outputFilenameTest, dataTest);
                     
                     
     }catch(Exception e){
         e.printStackTrace();
     }
     
     //write test into file
     
     //perform train+validation training /test evaluation
     
     predictiveSettings = new SettingsReader();
     numAttributes = dataTrainValidation.numAttributes();
     
      if(appset.system.equals("windows")){//createPredictiveSettingsFunction
             predictiveSettings.setPath(appset.outFolderPath+"\\predictiveOrigFinFC.s");//replace view1 with real name
            predictiveSettings.createPredictiveSettings(outputFilenameTrainValidation, outputFilenameTest, numAttributes, 1, 600, 0);

           //  predictiveSettings.setDataFilePath(appset.outFolderPath+"\\JinputInitial.arff");
     }
     else{
             predictiveSettings.setPath(appset.outFolderPath+"/predictiveOrigFinFC.s");
               predictiveSettings.createPredictiveSettings(outputFilenameTrainValidation, outputFilenameTest, numAttributes, 1, 600, 0);

     }
     
     //call CLUS
      exec.run(appset.javaPath,appset.clusPath ,appset.outFolderPath,"predictiveOrigFinFC.s",0, appset.clusteringMemory);
     
      //call weka models for FC features
       clsf = new J48();
      dataTrainValidation.deleteAttributeAt(0);
      dataTest.deleteAttributeAt(0);
      try{
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
            /*System.out.println("AUPRC0: "+eval.areaUnderPRC(0));
            System.out.println("AUC0: "+eval.areaUnderROC(0));
            System.out.println("AUPRC1: "+eval.areaUnderPRC(1));
            System.out.println("AUC1: "+eval.areaUnderROC(1));*/
            System.out.println("Av.AUPRC J48: "+auprc);
            System.out.println("Av.AUC J48: "+auc);
            System.out.println(eval.toSummaryString("\nResults OrigFC\n=====\n",false));
            
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
            System.out.println(eval.toSummaryString("\nResults OrigFC\n=====\n",false));
            
             if(!resultsWeka.containsKey("NB")){
                resultsWeka.put("NB", new ArrayList<>());
            }
            
            resultsWeka.get("NB").add(auprc);
             resultsWeka.get("NB").add(auc);
            
            clsf = new Logistic();
            
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
            System.out.println("Av.AUPRC Log: "+auprc);
            System.out.println("Av.AUC Log: "+auc);
            System.out.println(eval.toSummaryString("\nResults OrigFC\n=====\n",false));
            
            
             if(!resultsWeka.containsKey("Log")){
                resultsWeka.put("Log", new ArrayList<>());
            }
            
            resultsWeka.get("Log").add(auprc);
             resultsWeka.get("Log").add(auc);
            
            clsf = new MultilayerPerceptron();
            
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
            System.out.println("Av.AUPRC MLP: "+auprc);
            System.out.println("Av.AUC MLP: "+auc);
            System.out.println(eval.toSummaryString("\nResults OrigFC\n=====\n",false));
            
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
            System.out.println(eval.toSummaryString("\nResults OrigFC\n=====\n",false));
            
            
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
            System.out.println(eval.toSummaryString("\nResults OrigFC\n=====\n",false));
            
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
            System.out.println(eval.toSummaryString("\nResults OrigFC\n=====\n",false));
            
             if(!resultsWeka.containsKey("LMT")){
                resultsWeka.put("LMT", new ArrayList<>());
            }
            
            resultsWeka.get("LMT").add(auprc);
             resultsWeka.get("LMT").add(auc);
      }
      catch(Exception e){
          e.printStackTrace();
      }
      
      System.out.println("Original features completed!");
             
      //add all results from PCTRF
      try{
             Path p= null;
              BufferedReader read = null;
              String line = "";
              
               if(appset.system.equals("windows"))
                p= Paths.get(appset.outFolderPath+"\\predictiveAllRules.out");
              else  p= Paths.get(appset.outFolderPath+"/predictiveAllRules.out");
              
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
              
            resultsWeka.put("RF600PCT", new ArrayList<>());
              resultsWeka.get("RF600PCT").add(auprc);
              resultsWeka.get("RF600PCT").add(auc);   
              
              
           //   predictiveFinFCR
           
            if(appset.system.equals("windows"))
                p= Paths.get(appset.outFolderPath+"\\predictiveFinFCR.out");
              else  p= Paths.get(appset.outFolderPath+"/predictiveFinFCR.out");
              
              read = Files.newBufferedReader(p);
              line ="";
              
               perfLine = 0;
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
              
              resultsWeka.get("RF600PCT").add(auprc);
              resultsWeka.get("RF600PCT").add(auc);
              
               if(appset.system.equals("windows"))
                p= Paths.get(appset.outFolderPath+"\\predictiveAll.out");
              else  p= Paths.get(appset.outFolderPath+"/predictiveAll.out");
              
              read = Files.newBufferedReader(p);
              line ="";
              
              perfLine = 0;
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
              
              resultsWeka.get("RF600PCT").add(auprc);
              resultsWeka.get("RF600PCT").add(auc);
              
              //feature construction all
              
               if(appset.system.equals("windows"))
                p= Paths.get(appset.outFolderPath+"\\predictiveFinFC.out");
              else  p= Paths.get(appset.outFolderPath+"/predictiveFinFC.out");
              
              read = Files.newBufferedReader(p);
              line ="";
              
              perfLine = 0;
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
              
              resultsWeka.get("RF600PCT").add(auprc);
              resultsWeka.get("RF600PCT").add(auc);
              
              //orig features
               if(appset.system.equals("windows"))
                p= Paths.get(appset.outFolderPath+"\\predictiveOrig.out");
              else  p= Paths.get(appset.outFolderPath+"/predictiveOrig.out");
              
              read = Files.newBufferedReader(p);
              line ="";
              
              perfLine = 0;
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
              
              resultsWeka.get("RF600PCT").add(auprc);
              resultsWeka.get("RF600PCT").add(auc);
              
              //orig features FC
               if(appset.system.equals("windows"))
                p= Paths.get(appset.outFolderPath+"\\predictiveOrigFinFC.out");
              else  p= Paths.get(appset.outFolderPath+"/predictiveOrigFinFC.out");
              
              read = Files.newBufferedReader(p);
              line ="";
              
              perfLine = 0;
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
              
              resultsWeka.get("RF600PCT").add(auprc);
              resultsWeka.get("RF600PCT").add(auc);
              
      }
      catch(IOException e){
          e.printStackTrace();
      }
             
             String oWRes = "";
             if(appset.system.equals("windows"))
                        oWRes=appset.outFolderPath+"\\wekaComp.out";
             else oWRes = appset.outFolderPath+"/wekaComp.out";
             
             FileWriter fw = new FileWriter(oWRes);
             
             fw.write("FeatRulesAllAUPRC FeatRulesSelAUPRC FeatRulesRedsAllAUPRC FeatRulesRedsSelAUPRC FeatAllAUPRC FeatSelAUPRC FeatRulesAllAUC FeatRulesSelAUC FeatRulesRedsAllAUC FeatRulesRedsSelAUC FeatAllAUC FeatSelAUC\n\n");
             
             Iterator<String> it1 = resultsWeka.keySet().iterator();
             
             while(it1.hasNext()){
                 String a = it1.next();
                 fw.write(a+" "+resultsWeka.get(a).get(0)+" "+resultsWeka.get(a).get(2)+" "+resultsWeka.get(a).get(4)+" "+resultsWeka.get(a).get(6)+" "+resultsWeka.get(a).get(8)+" "+resultsWeka.get(a).get(10)+" "+resultsWeka.get(a).get(1)+" "+resultsWeka.get(a).get(3)+" "+resultsWeka.get(a).get(5)+" "+resultsWeka.get(a).get(7)+" "+resultsWeka.get(a).get(9)+" "+resultsWeka.get(a).get(11)+"\n");
             }
             fw.close();
      }
      catch(Exception e){
          e.printStackTrace();
      }
      
      //divide trainValidation to train + validation
      
      //perform feature selection procedure
      
      //create datasets with a given number of features
      
      //compute CLUS performance
      
      //add different classifiers from weka
      
       //add codes, and feature selection, prediction, construcion here
      //use always resSets.get(0)
      //filter redundant views, check if same-view used
      //rr1, rr
     //ucitati dataset s ciljnim atributom (samo jedan arrf, ne joined) - koristiti weku
      
     /* if(appset.system.equals("windows"))//write new file to disk
                        outputFilename= appset.outFolderPath+"\\JinputTest.arff";
                    else outputFilename= appset.outFolderPath+"/JinputTest.arff";
                     DataSink.write(outputFilename, dataTest);*/
      
      
     /* SettingsReader initSettings=new SettingsReader();//create settings for CLUS
        initSettings.setDataFilePath(appset.outFolderPath+"\\JinputInitial.arff");
        if(appset.system.equals("windows"))
             initSettings.setPath(appset.outFolderPath+"\\view1.s");
        else
             initSettings.setPath(appset.outFolderPath+"/view1.s");*/
      
      /*ClusProcessExecutor exec=new ClusProcessExecutor();

        //RunPredictiveFile
        exec.run(appset.javaPath,appset.clusPath ,appset.outFolderPath,"view1.s",0, appset.clusteringMemory);
      */
      
      //create a class to store evaluation results
      
   }
      else if(appset.exhaustiveTesting==1){
          System.out.println("type of experimentation: "+appset.exhaustiveTesting);
          for(int type=appset.parameters.get(2).get(0).intValue();type<=appset.parameters.get(2).get(1).intValue();type++){
              for(double minjs=appset.parameters.get(0).get(0);minjs<=appset.parameters.get(0).get(1);minjs+=appset.parameters.get(0).get(2)){
                  for(int minSupp=appset.parameters.get(1).get(0).intValue();minSupp<=appset.parameters.get(1).get(1).intValue();minSupp+=appset.parameters.get(1).get(2).intValue()){
                       appset.minJS=minjs;
          appset.minSupport= minSupp;
           Result=new RedescriptionSet();
           
      //ArrayList<RedescriptionSet> resSets=Result.createRedescriptionSetsCoocGen(rs,appset.preferences,type, appset,datJ,fid,coc);
           ArrayList<RedescriptionSet> resSets = null;
           if(datJ.numExamples<10000 && datJ.schema.getNbAttributes()-1<10000)
            resSets=Result.createRedescriptionSetsCoocGenPred(rs,appset.preferences,type, appset,datJ,fid,coc);//adds the most specific redescription first
      else
          resSets=Result.createRedescriptionSetsRandGen(rs,appset.preferences,type, appset,datJ,fid,coc);//should add one highly accurate redescription at random
      

      for(int rset=0;rset<resSets.size();rset++)
            resSets.get(rset).computeLift(datJ, fid);
  
     for(int fit=0;fit<resSets.size();fit++){
       coverage=new double[2];

      ResultsScore=resSets.get(fit).computeRedescriptionSetScoreGen(appset.preferences.get(fit),type,coverage,datJ,appset,fid);
     // resSets.get(fit).adaptSet(datJ, fid, 0);
      numFullRed=resSets.get(fit).computePVal(datJ,fid);
      System.out.println("Results score: "+ResultsScore);

      if(appset.system.equals("windows"))
        resSets.get(fit).writeToFile(appset.outFolderPath+"\\"+appset.outputName+"StLev_"+fit+" minjs "+appset.minJS+" JSType "+type+"minSupp "+appset.minSupport+".rr", datJ, fid, startTime,numFullRed,appset, ResultsScore, coverage,oom);
      else
          resSets.get(fit).writeToFile(appset.outFolderPath+"/"+appset.outputName+"StLev_"+fit+" minjs "+appset.minJS+" JSType "+type+"minSupp "+appset.minSupport+".rr", datJ, fid, startTime,numFullRed,appset, ResultsScore, coverage,oom);
      //resSets.get(fit).writePlots(appset.outFolderPath+"\\"+"RuleData"+"StLev_"+fit+" minjs "+appset.minJS+"JSType "+type+"minSupp "+appset.minSupport+".csv", appset,datJ,fid);
     
     //resSets.get(fit).redescriptions.clear();
     }
                  }
              }
          }
          
      }
     
      }
      
      
     /* coc.init(datJ.numExamples, datJ.schema.getNbAttributes()-1);
      coc.computeMatrix(Result, datJ);
      out=new File(appset.outFolderPath+"\\AttributesOpt.txt");
      coc.writeToFileAttributes(out, datJ.schema.getNbAttributes()-1);
      out=new File(appset.outFolderPath+"\\ElementsOpt.txt");
      coc.writeToFileElements(out, datJ.numExamples);*/
      
      Result.redescriptions.clear();
      rs.redescriptions.clear();
      rr.rules.clear();
      rr1.rules.clear();
      
      FileDeleter del=new FileDeleter();
     if(appset.system.equals("windows")){  
      del.setPath(appset.outFolderPath+"\\Jinputnew.arff");
      del.delete();
      del.setPath(appset.outFolderPath+"\\Jinputnew1.arff");
      del.delete();
      del.setPath(appset.outFolderPath+"\\Jinput.arff");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view1tmp.s");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view2tmp.s");
      del.delete();
       del.setPath(appset.outFolderPath+"\\view3tmp.s");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view1tmp1.s");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view2tmp1.s");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view1tmp.out");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view1tmp.model");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view2tmp.out");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view2tmp.model");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view1tmp1.out");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view1tmp1.model");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view2tmp1.out");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view2tmp1.model");
      del.delete();
      ///
      del.setPath(appset.outFolderPath+"\\view1tmpF1.s");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view2tmpF1.s");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view1tmpF.out");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view1tmpF.model");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view2tmpF.out");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view2tmpF.model");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view1tmpF1.out");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view1tmpF1.model");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view2tmpF1.out");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view2tmpF1.model");
      del.delete();
      ///
      del.setPath(appset.outFolderPath+"\\view3tmp.out");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view3tmp.model");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view1.s");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view1.out");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view1.model");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view2.s");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view2.out");
      del.delete();
      del.setPath(appset.outFolderPath+"\\view2.model");
      del.delete();
     }
     else{
       del.setPath(appset.outFolderPath+"/Jinputnew.arff");
      del.delete();
      del.setPath(appset.outFolderPath+"/Jinputnew1.arff");
      del.delete();
      del.setPath(appset.outFolderPath+"/Jinput.arff");
      del.delete();
      del.setPath(appset.outFolderPath+"/view1tmp.s");
      del.delete();
      del.setPath(appset.outFolderPath+"/view2tmp.s");
      del.delete();
       del.setPath(appset.outFolderPath+"/view3tmp.s");
      del.delete();
      del.setPath(appset.outFolderPath+"/view1tmp1.s");
      del.delete();
      del.setPath(appset.outFolderPath+"/view2tmp1.s");
      del.delete();
      del.setPath(appset.outFolderPath+"/view1tmp.out");
      del.delete();
      del.setPath(appset.outFolderPath+"/view1tmp.model");
      del.delete();
      del.setPath(appset.outFolderPath+"/view2tmp.out");
      del.delete();
      del.setPath(appset.outFolderPath+"/view2tmp.model");
      del.delete();
      del.setPath(appset.outFolderPath+"/view1tmp1.out");
      del.delete();
      del.setPath(appset.outFolderPath+"/view1tmp1.model");
      del.delete();
      del.setPath(appset.outFolderPath+"/view2tmp1.out");
      del.delete();
      del.setPath(appset.outFolderPath+"/view2tmp1.model");
      del.delete();
      ///
      del.setPath(appset.outFolderPath+"/view1tmpF1.s");
      del.delete();
      del.setPath(appset.outFolderPath+"/view2tmpF1.s");
      del.delete();
      del.setPath(appset.outFolderPath+"/view1tmpF.out");
      del.delete();
      del.setPath(appset.outFolderPath+"/view1tmpF.model");
      del.delete();
      del.setPath(appset.outFolderPath+"/view2tmpF.out");
      del.delete();
      del.setPath(appset.outFolderPath+"/view2tmpF.model");
      del.delete();
      del.setPath(appset.outFolderPath+"/view1tmpF1.out");
      del.delete();
      del.setPath(appset.outFolderPath+"/view1tmpF1.model");
      del.delete();
      del.setPath(appset.outFolderPath+"/view2tmpF1.out");
      del.delete();
      del.setPath(appset.outFolderPath+"/view2tmpF1.model");
      del.delete();
      ///
      del.setPath(appset.outFolderPath+"/view3tmp.out");
      del.delete();
      del.setPath(appset.outFolderPath+"/view3tmp.model");
      del.delete();
      del.setPath(appset.outFolderPath+"/view1.s");
      del.delete();
      del.setPath(appset.outFolderPath+"/view1.out");
      del.delete();
      del.setPath(appset.outFolderPath+"/view1.model");
      del.delete();
      del.setPath(appset.outFolderPath+"/view2.s");
      del.delete();
      del.setPath(appset.outFolderPath+"/view2.out");
      del.delete();
      del.setPath(appset.outFolderPath+"/view2.model");
      del.delete();  
     }
    
 }
}
