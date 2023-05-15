/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package redescriptionmining;

import clus.data.io.ARFFFile;
import clus.data.io.ClusReader;
import clus.data.rows.RowData;
import clus.main.Settings;
import gnu.trove.iterator.TIntIterator;
import java.io.BufferedReader;
import java.util.ArrayList;
import java.util.Collections;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Iterator;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/**
 *
 * @author matej
 */


public class Main {

    /**
     * @param args the command line arguments
     */
    
    
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
       //testing
        ApplicationSettings appset=new ApplicationSettings();
        Mappings fid=new Mappings();
        fid.createIndex("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\UNCTADall3test.arff");
        DataSetCreator dat=new DataSetCreator("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\UNCTADall3test.arff");
        try{
        dat.readDataset();
        }
        catch(Exception e){
            e.printStackTrace();
        }
        System.out.println();
       
        System.out.println("value: "+dat.getValue(fid.attId.get("E9"),fid.exampleId.get("\"210\"") /*184*/));
        System.out.println("value: "+dat.getValue(fid.attId.get("E1"),fid.exampleId.get("\"210\"") /*184*/));
        System.out.println("value: "+dat.getValue(fid.attId.get("E2"),fid.exampleId.get("\"210\"") /*184*/));
        System.out.println("value: "+dat.getValue(fid.attId.get("E3"),fid.exampleId.get("\"210\"") /*184*/));
        
        
        ProcessBuilder pb = new ProcessBuilder("C:\\Program Files\\Java\\jdk1.7.0_40\\bin\\java.exe", "-jar", "C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\Clus\\Clus.jar" , "-rules","unctad.s");
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
         
        pb = new ProcessBuilder("C:\\Program Files\\Java\\jdk1.7.0_40\\bin\\java.exe", "-jar", "C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\Clus\\Clus.jar", "-rules","wb.s");
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
        
        //reading arff file
        /* String ARFF_FILE_PATH = "YOUR_ARFF_FILE_PATH";
        ArffLoader arffLoader = new ArffLoader();

        File datasetFile = new File(ARFF_FILE_PATH);
        Instances dataInstances=null;
        try{
        arffLoader.setFile(datasetFile);

         dataInstances = arffLoader.getDataSet();
        }catch(Exception e){
            e.printStackTrace();
            return;
        }
        */
        String input="C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\wb.out";
        String input1="C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\unctad.out";
        RuleReader rr=new RuleReader();
        rr.extractRules(input,fid,dat,appset);
        RuleReader rr1=new RuleReader();
        rr1.extractRules(input1,fid,dat,appset);
        
        /*System.out.println("Rules found: ");
        for(int i=0;i<rr.rules.size();i++){
            System.out.println("Rule: "+rr.rules.get(i).rule);
            System.out.println();
            System.out.println("Elements: ");
            System.out.println();
            
            for(String s:rr.rules.get(i).elements)
                System.out.println(s);
        }*/
        
        Jacard js=new Jacard();
       // System.out.println("similarity: "+js.computeJacard(rr.rules.get(0), rr.rules.get(1)));
        
        ArrayList<Redescription> redescriptions=new ArrayList<Redescription>();
        
        System.out.println("max number of rules: "+(rr.rules.size()*rr1.rules.size()));
        int numIt=0,maxNum=rr.rules.size()*rr1.rules.size();
        int step=maxNum/100;
        
        for(int i=0;i<rr.rules.size();i++)
            for(int j=0;j<rr1.rules.size();j++){
                js.initialize();
                numIt++;
                if(js.computeJacard(rr.rules.get(i), rr1.rules.get(j))>0){
                    Redescription tmp=new Redescription(rr.rules.get(i).rule,rr1.rules.get(j).rule,js.JS,fid,dat);
                    tmp.computeElements(rr.rules.get(i), rr1.rules.get(j));
                    tmp.computeUnion(rr.rules.get(i), rr1.rules.get(j));
                    
                    int found=0;
                   // System.out.println("Broj pravila: "+redescriptions.size());
                    for(int k=0;k<redescriptions.size();k++){
                        int quality=tmp.CompareQuality(redescriptions.get(k));
                         found=0;
                        if(quality==-1){
                            continue;
                        }
                        else if(quality==-2){
                         found=1;
                            break;
                        }
                        else if(quality==2){
                           
                           // tmp.join(redescriptions.get(k),rr,rr1 rr.rules.get(tmp.LSIndex), rr1.rules.get(tmp.RSIndex), rr.rules.get(redescriptions.get(k).LSIndex), rr1.rules.get(redescriptions.get(k).RSIndex));
                           // tmp.join(redescriptions.get(k),fid,dat);
                            redescriptions.set(k, tmp);
                            found=1;
                            break;
                        }
                        else if(quality==1){
                            redescriptions.set(k, tmp);
                            found=1;
                            break;
                        }
                            }
                    if(found==0)
                    redescriptions.add(tmp);  
                }
                if(numIt%step==0)
                System.out.println((((double)numIt/maxNum)*100)+"% completed...");
                if(numIt==maxNum)
                    System.out.println("100% completed!");
            }
        
        System.out.println("Normalizing rules: ");
        for(int i=0;i<redescriptions.size();i++){
            String left=normalize(redescriptions.get(i).ruleStrings.get(0));
            redescriptions.get(i).ruleStrings.add(left);
            String right=normalize(redescriptions.get(i).ruleStrings.get(1));
            redescriptions.get(i).ruleStrings.add(right);
        }
        System.out.println("Normalization complete!");
        
        System.out.println("Sorting rules!");
        Collections.sort(redescriptions,Collections.reverseOrder());
        
                       try {
        BufferedWriter bw;
        File file = new File("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\redescriptionsF.rr");
        
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
         numIt=0;
         maxNum=redescriptions.size();
         step=maxNum/100;
         
         System.out.println("Number of redescriptions: "+maxNum);
         
        for(int i=0;i<redescriptions.size();i++){
            bw.write("Rules: \n\n");
            //System.out.println("LSR: "+redescriptions.get(i).LSrule);
            bw.write("LSR: "+redescriptions.get(i).ruleStrings.get(0)+"\n");
           // System.out.println("RSR: "+redescriptions.get(i).RSrule);
            bw.write("RSR: "+redescriptions.get(i).ruleStrings.get(1)+"\n");
           // System.out.println("JS: "+redescriptions.get(i).JS);
             bw.write("JS: "+redescriptions.get(i).JS+"\n");
             bw.write("Support intersection: "+redescriptions.get(i).elements.size()+"\n");
             bw.write("Support union: "+redescriptions.get(i).elementsUnion.size()+"\n\n");
            //System.out.println("Covered examples: ");
             bw.write("Covered examples (intersection): \n");
             
             TIntIterator it=redescriptions.get(i).elements.iterator();
             
             while(it.hasNext()){
                 int s=it.next();
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
            
            it=redescriptions.get(i).elementsUnion.iterator();
            
            while(it.hasNext()){
                int s=it.next();
                bw.write(fid.idExample.get(s)+" ");
            }
            
           /* for(int s:redescriptions.get(i).elementsUnion){
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
