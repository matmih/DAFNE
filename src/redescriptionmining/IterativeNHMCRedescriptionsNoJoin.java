/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package redescriptionmining;

import clus.data.io.ARFFFile;
import static clus.data.io.ARFFFile.writeArffHeader;
import clus.data.io.ClusReader;
import clus.data.rows.DataTuple;
import clus.data.rows.RowData;
import clus.data.type.ClusAttrType;
import clus.data.type.ClusSchema;
import clus.data.type.NumericAttrType;
import clus.main.Settings;
import clus.util.ClusException;
import gnu.trove.iterator.TIntIterator;
import java.io.BufferedReader;
import java.util.ArrayList;
import java.util.Collections;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Random;
import jeans.util.IntervalCollection;
import org.apache.commons.math3.distribution.BinomialDistribution;

/**
 *
 * @author matej
 */


public class IterativeNHMCRedescriptionsNoJoin {

    /**
     * @param args the command line arguments
     */
    
    public static void writeArff(String fname, RowData data) throws IOException, ClusException {
		PrintWriter wrt = new PrintWriter(new OutputStreamWriter(new FileOutputStream(fname)));
		ClusSchema schema = data.getSchema();
		writeArffHeader(wrt, schema);
		wrt.println("@DATA");
		for (int j = 0; j < data.getNbRows(); j++) {
			DataTuple tuple = data.getTuple(j);
                        wrt.write(tuple.m_Objects[0]+",");
                        for(int k=0;k<tuple.m_Doubles.length;k++)
                            if(k+1<tuple.m_Doubles.length)
                                if(Double.isInfinite(tuple.m_Doubles[k]))
                                    wrt.write("?"+",");
                                else
                                     wrt.write(""+tuple.m_Doubles[k]+",");
                        else
                                if(Double.isInfinite(tuple.m_Doubles[k]))
                                     wrt.write("?"+"\n");
                                else
                                    wrt.write(""+tuple.m_Doubles[k]+"\n");    
			//tuple.writeTuple(wrt);
		}
		wrt.close();
	}
    
    
      public static String normalize(String rule){
        
        HashMap<String,ArrayList<Double>> leftCM=new HashMap<String,ArrayList<Double>>();
        
        String rs[]=rule.split(" AND ");
        
        for(int i=0;i<rs.length;i++){
                if(rs[i].contains(">")){
                    String t[]=rs[i].split(" > ");
                    if(!leftCM.containsKey(t[0])){
                        ArrayList<Double> tmp=new ArrayList<Double>(Collections.nCopies(4, 0.0));
                        t[1]=t[1].replaceAll(",", ".");
                        tmp.set(0,Double.parseDouble(t[1]));
                        tmp.set(1,1.0);
                        tmp.set(2, Double.MAX_VALUE);
                        leftCM.put(t[0], tmp);
                    }
                    else{
                        ArrayList<Double> tmp=leftCM.get(t[0]);
                        t[1]=t[1].replaceAll(",", ".");
                        Double value=Double.parseDouble(t[1]);
                        tmp.set(1, 1.0);
                        if(value>tmp.get(0)){
                            tmp.set(0,value);
                            leftCM.put(t[0], tmp);
                        }
                    }
                }
                else if(rs[i].contains("<=")){
                     String t[]=rs[i].split(" <= ");
                    if(!leftCM.containsKey(t[0])){
                       ArrayList<Double> tmp=new ArrayList<Double>(Collections.nCopies(4, 0.0));
                        t[1]=t[1].replaceAll(",", ".");
                        tmp.set(2,Double.parseDouble(t[1]));
                        tmp.set(3,-1.0);
                        leftCM.put(t[0], tmp);
                    }
                    else{
                       ArrayList<Double> tmp=leftCM.get(t[0]);
                       t[1]=t[1].replaceAll(",", ".");
                        Double value=Double.parseDouble(t[1]);
                        tmp.set(3,-1.0);
                        if(value<tmp.get(2)){
                            tmp.set(2, value);
                            leftCM.put(t[0], tmp); 
                        }
                    }
                }
            }
        
        Iterator<String> it=leftCM.keySet().iterator();
        
        String rule1="";
        
        while(it.hasNext()){
            int haslR=0;
           String var=it.next();
           ArrayList<Double> tmp=leftCM.get(var);
            rule1+=var+" ";
            if(tmp.get(1)==1.0){
                rule1+="> ";
            rule1+=tmp.get(0);
            haslR=1;
            }
             if(tmp.get(3)==-1.0){
                 if(haslR==1)
                     rule1+=" ";
                 rule1+="<= ";
                 rule1+=tmp.get(2);
             }
            if(it.hasNext())
                rule1+=" AND ";
        }
        return rule1;
    }
    
    public static void main(String[] args) {
        // TODO code application logic here
       /*int test=1;
        {//test phase
        
        SettingsReader set=new SettingsReader("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\wbtmp.s","C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\wbtmpS.s");
        set.ModifySettings(50);
        Settings cset=new Settings();
        
        ClusReader cread=null;
        try{
        cread=new ClusReader("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\UNCTADall3test.arff",cset);
        }
        catch(IOException io){
            io.printStackTrace();
        }
        ARFFFile a=new ARFFFile(cread);
        RowData data=null;
        ClusSchema schema=null;
        try{
        data=a.readArff("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\UNCTADall3test.arff");
        schema=data.getSchema();
        }
        catch(Exception e){
           e.printStackTrace();
        }
        System.out.println(data);
        System.out.println("Attributes: ");
        System.out.println(schema.getNbAttributes());
        int nbAttr=schema.getNbAttributes();
       // ClusAttrType at=new ClusAttrType("name","type");
        schema.getAttrType(0).setStatus(ClusAttrType.STATUS_KEY);
        for (int iAttr = 0; iAttr < nbAttr ; iAttr++) {
				ClusAttrType attrType = schema.getAttrType(iAttr);
                                if(attrType.getStatus()==ClusAttrType.STATUS_KEY)
                                    System.out.println(attrType.getName()+" "+"key");
                                else
                                System.out.println(attrType.getName()+" "+attrType.getTypeName().toLowerCase());
        }
        
        int nattr=schema.getNbAttributes();
        
        for(int i=0;i<50;i++)
            schema.addAttrType(new NumericAttrType("target"+(i+1)));
        schema.setClusteringAll(true);

        System.out.println("New attributes: ");
        for (int iAttr = 0; iAttr < schema.getNbAttributes() ; iAttr++) {
				ClusAttrType attrType = schema.getAttrType(iAttr);
                                if(attrType.getStatus()==ClusAttrType.STATUS_KEY)
                                    System.out.println(attrType.getName()+" "+"key");
                                else
                                System.out.println(attrType.getName()+" "+attrType.getTypeName().toLowerCase());
        }
        
        ArrayList<DataTuple> dataList=data.toArrayList();
        DataTuple asd=dataList.get(0);
        
        Random rand=new Random();
        for (int j=0;j<data.getNbRows();j++){
        double arow[]=new double[schema.getNbAttributes()-1];
        for(int k=0;k<dataList.get(j).m_Doubles.length;k++){
            arow[k]=dataList.get(j).m_Doubles[k];
                    }
            for(int i=0;i<50;i++)
                arow[dataList.get(j).m_Doubles.length+i]=rand.nextDouble();
            dataList.get(j).m_Doubles=arow;
        }
        
        System.out.println("nattributes before: "+nattr);
        System.out.println("nattributes after: "+schema.getNbAttributes());
        System.out.println("size dataarray: "+dataList.size());
        System.out.println("size of the objects in first row: "+dataList.get(0).m_Objects.length);
         System.out.println("size of the doubles in first row: "+dataList.get(0).m_Doubles.length);
       data.setFromList(dataList);
       data.setSchema(schema);
       schema.setSettings(cset);
       
       System.out.println("nbrows: "+data.getNbRows());
       
      /* for(int j=0;j<data.getNbRows();j++) 
        for(int i=0;i<schema.getNbAttributes();i++)
            if(i<nattr)
            joinList.add(dataList.get(i));
            else joinList.add(addon.get(i-nattr));
       data.setFromList(joinList);*/
       
       /*File newArff=new File("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\UNCTADall3testnew.arff");
       PrintWriter wrt=null;
       try{
             wrt=new PrintWriter(newArff);
       }
       catch(FileNotFoundException io){
           io.printStackTrace();
       }
       try{
      // writeArffHeader(wrt, schema);
       writeArff("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\UNCTADall3testnew.arff", data);
       }
       catch(Exception e){
           e.printStackTrace();
       }
       
      /* for(int i=0;i<data.getNbRows();i++)
           for(int j=0;j<50;j++)
           data.getTuple(i).m_Doubles*/
        /*try{
         BufferedReader reader = new BufferedReader(
                              new FileReader("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\UNCTADall3test.arff"));
 Instances data = new Instances(reader);
 reader.close();
 System.out.println(data);
        }
        catch(IOException e){
            e.printStackTrace();
        }*/
        
      /*  if(test==1)
        return;
        }*/
        ApplicationSettings appset=new ApplicationSettings();
        int minSupport=2; 
        double minJS=0.6;
        Mappings fid=new Mappings();
        fid.createIndex("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\UNCTADall3test.arff");
        DataSetCreator dat=new DataSetCreator("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\UNCTADall3test.arff");
        try{
        dat.readDataset();
        }
        catch(Exception e){
            e.printStackTrace();
        }
        
        ProcessBuilder pb = new ProcessBuilder("C:\\Program Files\\Java\\jdk1.8.0_25\\bin\\java.exe", "-jar", "C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\Clus\\dist\\CLUSNHMC.jar" , "-forest","unctadNHMCf.s");
        pb.directory(new File("C:\\Users\\matej\\Documents\\Redescription mining with CLUS"));
        pb.redirectErrorStream(true); 
        Process p =null;
        try{
            p= pb.start();
                
            InputStreamReader isr = new  InputStreamReader(p.getInputStream());
            BufferedReader br = new BufferedReader(isr);

            String lineRead;
            while ((lineRead = br.readLine()) != null) {
                 // swallow the line, or print it out - System.out.println(lineRead);
                   System.out.println(lineRead);
            }
        }catch(java.io.IOException e){
            e.printStackTrace();
            System.exit(-1);
        }
         try{
        p.waitFor();
        }
        catch(java.lang.InterruptedException e1){ 
            e1.printStackTrace();
        }
        
         System.out.println("Process 1 finished!");
         
        //read the rules obtained from first attribute set
          String input1="C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\unctadNHMCf.out";
           RuleReader rr1=new RuleReader();
           rr1.extractRules(input1,fid,dat,appset);
           
         //reading arff file
        
        //read the .arff file
        //read the setting2.s file
           
        SettingsReader set=new SettingsReader("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\unctadNHMCf.s","C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\unctadNHMCf.s");
       //set.setDataFilePath("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\UNCTADall3testnew.arff");
        set.changeAlpha(0.5);
       /*Settings cset=new Settings(); 
        ClusReader cread=null;
        try{
        cread=new ClusReader("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\UNCTADall3test.arff",cset);
        }
        catch(IOException io){
            io.printStackTrace();
        }
        ARFFFile a=new ARFFFile(cread);
        RowData data=null;
        ClusSchema schema=null;
        try{
        data=a.readArff("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\UNCTADall3test.arff");
        schema=data.getSchema();
        }
        catch(Exception e){
           e.printStackTrace();
        }
           int naex=data.getNbRows();
           int nARules=0;
        //locate the percentage of the rules to look at (that are specific enough) 
        //change the target label in the setting2.s file and save it
        //create a new .arff file with the corresponding target attributes
           
            ArrayList<DataTuple> dataList=data.toArrayList();
        
            for(int i=0;i<rr1.rules.size();i++)
                if(rr1.rules.get(i).elements.size()<=naex*0.2)
                    nARules++;
          
            for(int i=0;i<nARules;i++)
            schema.addAttrType(new NumericAttrType("target"+(i+1)));
        //schema.setClusteringAll(true);
            System.out.println("nARules: "+nARules);
            System.out.println("New number of attributes: "+schema.getNbAttributes());
        //Random rand=new Random();
         int count=0;   
            
        for (int j=0;j<data.getNbRows();j++){
            count=0;
        double arow[]=new double[schema.getNbAttributes()-1];
        for(int k=0;k<dataList.get(j).m_Doubles.length;k++){
            arow[k]=dataList.get(j).m_Doubles[k];
                    }
            for(int i=0;i<rr1.rules.size();i++)
                if(rr1.rules.get(i).elements.size()<=naex*0.2){
                    if(rr1.rules.get(i).elements.contains(dataList.get(j).m_Objects[0]))
                arow[dataList.get(j).m_Doubles.length+count-1]=1.0;
                    else
                         arow[dataList.get(j).m_Doubles.length+count-1]=0.0;
                    count++;
                }
            dataList.get(j).m_Doubles=arow;
        }
           
       data.setFromList(dataList);
       data.setSchema(schema);
       schema.setSettings(cset);
        set.ModifySettings(nARules);
       try{
      // writeArffHeader(wrt, schema);
       writeArff("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\UNCTADall3testnew.arff", data);
       }
       catch(Exception e){
           e.printStackTrace();
       }
       
       */
        int leftSide=1, rightSide=0;//set left to 1 when computing lf, otherwise right
        int it=0;
        Jacard js=new Jacard();
       // System.out.println("similarity: "+js.computeJacard(rr.rules.get(0), rr.rules.get(1)));
        
        ArrayList<Redescription> redescriptions=new ArrayList<Redescription>();
        RuleReader rr=new RuleReader();
        int newRedescriptions=1;
        int numIter=0;
        int RunInd=0, maxDistance=100;
         NHMCDistanceMatrix NHMCdist=new NHMCDistanceMatrix(dat.numExamples,appset);
        File distanceFile=new File("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\distances.csv");   
        
        while(newRedescriptions!=0 && RunInd<2){
       
             newRedescriptions=0;
            System.out.println("Iteration: "+(++numIter));
            
          
        RuleReader ItRules=new RuleReader();
      
        int naex=dat.numExamples;
      /* int nARules=0;
       
       if(leftSide==1){
       for(int i=rr1.newRuleIndex;i<rr1.rules.size();i++)
                if(rr1.rules.get(i).elements.size()<=naex*0.2 && rr1.rules.get(i).elements.size()>3)
                    nARules++;
       }
       else if(rightSide==1){
                 for(int i=rr.newRuleIndex;i<rr.rules.size();i++)
                if(rr.rules.get(i).elements.size()<=naex*0.2 && rr.rules.get(i).elements.size()>3)
                    nARules++;
       }
     */
        //run the second proces on new data
        // iterate until convergence (no new rules, or very small amount obtained)
        
         if(leftSide==1){   
       NHMCdist.computeDistanceMatrix(rr1, fid, maxDistance, dat.numExamples);
       NHMCdist.writeToFile(distanceFile,fid,appset);
       NHMCdist.reset(appset);
        pb = new ProcessBuilder("C:\\Program Files\\Java\\jdk1.8.0_25\\bin\\java.exe", "-jar", "C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\Clus\\dist\\CLUSNHMC.jar", "-forest","wbNHMCf.s");
        pb.directory(new File("C:\\Users\\matej\\Documents\\Redescription mining with CLUS"));
        try{
                p= pb.start();
                
                 InputStreamReader isr = new  InputStreamReader(p.getInputStream());
                BufferedReader br = new BufferedReader(isr);

                 String lineRead;
  while ((lineRead = br.readLine()) != null) {
    // swallow the line, or print it out - System.out.println(lineRead);
      System.out.println(lineRead);
  }
        }catch(java.io.IOException e){
            e.printStackTrace();
            System.exit(-1);
        }
         try{
        p.waitFor();
        }
        catch(java.lang.InterruptedException e1){ 
            e1.printStackTrace();
        }
        System.out.println("Process 2 finished!");
         }
         else if(rightSide==1){
             NHMCdist.computeDistanceMatrix(rr, fid, maxDistance, dat.numExamples);
             NHMCdist.writeToFile(distanceFile,fid,appset);
             NHMCdist.reset(appset);
            pb = new ProcessBuilder("C:\\Program Files\\Java\\jdk1.8.0_25\\bin\\java.exe", "-jar", "C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\Clus\\dist\\CLUSNHMC.jar", "-forest","unctadNHMCf.s");
            pb.directory(new File("C:\\Users\\matej\\Documents\\Redescription mining with CLUS"));
        try{
                p= pb.start();
                
                 InputStreamReader isr = new  InputStreamReader(p.getInputStream());
                BufferedReader br = new BufferedReader(isr);

                 String lineRead;
  while ((lineRead = br.readLine()) != null) {
    // swallow the line, or print it out - System.out.println(lineRead);
      System.out.println(lineRead);
  }
        }catch(java.io.IOException e){
            e.printStackTrace();
            System.exit(-1);
        }
         try{
        p.waitFor();
        }
        catch(java.lang.InterruptedException e1){ 
            e1.printStackTrace();
        }
        System.out.println("Process 1 finished!"); 
         }
        /*Process proc=null;
        try{
             proc = Runtime.getRuntime().exec("java -jar Validate.jar");
        }
        catch(java.io.IOException e){
            e.printStackTrace();
            return;
        }
        try{
        proc.waitFor();
        }
        catch(java.lang.InterruptedException e1){ 
            e1.printStackTrace();
        }*/
        
        String input="";
        if(leftSide==1)
         input="C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\wbNHMCf.out";
        else if(rightSide==1)
            input="C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\unctadNHMCf.out";
        //String input1="C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\unctad.out";
       int newRules=0;
        ItRules.extractRules(input,fid,dat,appset);
        if(leftSide==1){
            newRules=rr.addnewRules(ItRules, 1);
       // rr.extractRules(input);
        }
        else if(rightSide==1){
            newRules=rr1.addnewRules(ItRules, 1);
            //rr1.extractRules(input);
        }
        
        System.out.println("New rules: "+newRules);
       // RuleReader rr1=new RuleReader();
       // rr1.extractRules(input1);
        
        /*System.out.println("Rules found: ");
        for(int i=0;i<rr.rules.size();i++){
            System.out.println("Rule: "+rr.rules.get(i).rule);
            System.out.println();
            System.out.println("Elements: ");
            System.out.println();
            
            for(String s:rr.rules.get(i).elements)
                System.out.println(s);
        }*/
        
        System.out.println("max number of rules: "+((rr.rules.size()-rr.newRuleIndex)*(rr1.rules.size()-rr1.newRuleIndex)));
        int numIt=0,maxNum=(rr.rules.size()-rr.newRuleIndex)*(rr1.rules.size()-rr1.newRuleIndex);
        int step=maxNum/100;
        for(int i=rr.newRuleIndex;i<rr.rules.size();i++){
           // long startTime = System.currentTimeMillis();
            for(int j=rr1.newRuleIndex;j<rr1.rules.size();j++){
                js.initialize();
                numIt++;
                if(js.computeJacard(rr.rules.get(i), rr1.rules.get(j))>0.1){
                    Redescription tmp=new Redescription(rr.rules.get(i).rule,rr1.rules.get(j).rule,js.JS,fid,dat);
                    tmp.computeElements(rr.rules.get(i), rr1.rules.get(j));
                    tmp.computeUnion(rr.rules.get(i), rr1.rules.get(j));
                    
                    
                        if(tmp.elements.size()>minSupport){//tmp.JS>0.4 || (tmp.elements.size()>1 && tmp.JS>0.2)
                    redescriptions.add(tmp);
                    newRedescriptions++;
                        }
                }
                if(numIt%step==0)
                System.out.println((((double)numIt/maxNum)*100)+"% completed...");
                if(numIt==maxNum)
                    System.out.println("100% completed!");
                it++;                 
            }
           // long stopTime = System.currentTimeMillis();
       // System.out.println("Time required for one rule complete: "+(stopTime-startTime));
        }
        System.out.println("New redescriptions: "+newRedescriptions);
        if(leftSide==1){
            leftSide=0;
            rightSide=1;
        }
        else if(rightSide==1){
            rightSide=0;
            leftSide=1;
        }
        RunInd++;
        System.out.println("Running index: "+RunInd);
        }
        //normalization after all iterations
       /*        System.out.println("Normalizing rules: ");
        for(int i=0;i<redescriptions.size();i++){
            String left=normalize(redescriptions.get(i).LSrule);
            redescriptions.get(i).LSrule=left;
            String right=normalize(redescriptions.get(i).RSrule);
            redescriptions.get(i).RSrule=right;
        }
        System.out.println("Normalization complete!");
        */
       
        //removing all redescriptions with inadequate minSupport and minJS
       for (Iterator<Redescription> iteratorR = redescriptions.iterator(); iteratorR.hasNext(); ) {
            Redescription test = iteratorR.next();
                if (test.elements.size()<minSupport || test.JS<minJS ) {
                     iteratorR.remove();
             }
            }
               

        //filtering
       System.out.println("Filtering redescriptions!");
       ArrayList<Redescription> toRemove=new ArrayList<>();
      while(true){
          int ok=0;
          int iterationCount=0;
          System.out.println("red size in filtering: "+redescriptions.size());
        for(int i=iterationCount;i<redescriptions.size();i++){
            ok=1;
            for(int j=i+1;j<redescriptions.size();j++){
                if(redescriptions.get(i).CompareEqual(redescriptions.get(j))==2){
//                    redescriptions.get(i).join(redescriptions.get(j), fid,dat);
                    toRemove.add(redescriptions.get(j));
                }
         }
            if(toRemove.size()==0)
                iterationCount=i;
            if(toRemove.size()>0){
                for(int j=0;j<toRemove.size();j++)
                    redescriptions.remove(toRemove.get(j));
                ok=0;
                toRemove.clear();
                break;
            }
       }
        System.out.println("Iteration count: "+iterationCount);
        if(ok==1)
            break;
     }
         
      int numFullRed=0;
        //closing intervals && computing pVal...
        for(int i=0;i<redescriptions.size();i++){
            redescriptions.get(i).closeInterval(dat, fid);
            double prob=((double)(redescriptions.get(i).supportsSides.get(0)*redescriptions.get(i).supportsSides.get(1)))/(dat.numExamples*dat.numExamples);
            BinomialDistribution dist=new BinomialDistribution(dat.numExamples,prob);
            redescriptions.get(i).pVal=1.0-dist.cumulativeProbability(redescriptions.get(i).elements.size());
            //System.out.println("Probability: "+dist.cumulativeProbability(2));
            if(redescriptions.get(i).JS==1.0)
                numFullRed++;
        }
        
        System.out.println("Found "+numFullRed+" redescriptions with JS=1.0 and minsupport>"+minSupport);
         NHMCdist.resetFile(distanceFile);
        set.changeAlpha(1.0);
        //sorting redescriptions
        System.out.println("Sorting rules!");
        Collections.sort(redescriptions,Collections.reverseOrder());
        

        //writing redescriptions to file
                       try {
        BufferedWriter bw;
        File file = new File("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\redescriptionsNHMCExperimentalIterativeNoJoin.rr");
        
        if (!file.exists()) {
				file.createNewFile();
			}
        FileWriter fw = new FileWriter(file.getAbsoluteFile());
			 bw= new BufferedWriter(fw);
			/*bw.write(output);
			bw.close();*/
 
			//System.out.println("Done");
                         
        
        //String output="";
        
        System.out.println();
        System.out.println();
        System.out.println("Redescriptions: ");
         bw.write("Redescriptions: \n\n");
         int numIt=0;
         int maxNum=redescriptions.size();
         int step=maxNum/100;
         
         if(maxNum<100)
             step=maxNum;
             
         System.out.println("Number of redescriptions: "+maxNum);
         
        for(int i=0;i<redescriptions.size();i++){
            redescriptions.get(i).createRuleString(fid);
            bw.write("Rules: \n\n");
            //System.out.println("LSR: "+redescriptions.get(i).LSrule);
            bw.write("LSR: "+redescriptions.get(i).ruleStrings.get(0)+"\n");
           // System.out.println("RSR: "+redescriptions.get(i).RSrule);
            bw.write("RSR: "+redescriptions.get(i).ruleStrings.get(1)+"\n");
           // System.out.println("JS: "+redescriptions.get(i).JS);
             bw.write("JS: "+redescriptions.get(i).JS+"\n");
             bw.write("p-value :"+redescriptions.get(i).pVal+"\n");
             bw.write("Support intersection: "+redescriptions.get(i).elements.size()+"\n");
             bw.write("Support union: "+redescriptions.get(i).elementsUnion.size()+"\n\n");
            //System.out.println("Covered examples: ");
             bw.write("Covered examples (intersection): \n");
             
             TIntIterator itT=redescriptions.get(i).elements.iterator();
            
            while(itT.hasNext()){
                int s=itT.next();
                bw.write(fid.idExample.get(s)+" ");
            }
             
            /*for(int s:redescriptions.get(i).elements){
               // System.out.print(s+" ");
                bw.write(fid.idExample.get(s)+" ");
                //output+=s+" ";
            }*/
            //output+="\n";
            bw.write("\n");
            bw.write("Union elements: \n");
            //System.out.println("Union: ");
            
            itT=redescriptions.get(i).elementsUnion.iterator();
            
            while(itT.hasNext()){
                int s=itT.next();
                bw.write(fid.idExample.get(s)+" ");
            }
            
            /*for(int s:redescriptions.get(i).elementsUnion){
               // System.out.print(s+" ");
                bw.write(fid.idExample.get(s)+" ");
                //output+=s+" ";
            }*/
            //System.out.println();
            //output+="\n\n";
            bw.write("\n\n");
            numIt++;
            if(numIt%step==0)
                System.out.println((((double)numIt/maxNum)*100)+"% completed...");
                if(numIt==maxNum)
                    System.out.println("100% completed!");
            
        }
            
        rr.rules.clear();
        rr1.rules.clear();
        redescriptions.clear();
                         
        bw.close();
    }
                       catch (IOException e) {
			e.printStackTrace();
		}
 }
}