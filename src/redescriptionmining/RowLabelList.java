/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package redescriptionmining;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Locale;

/**
 *
 * @author matej
 * creates redescriptions in readable format, row by row
 */
public class RowLabelList {
    static public void main(String args[]){
        
        File input=new File("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\Redescriptions+distributionCP\\redescriptionsGuidedExperimentalIterativeTSMJsupp100-300CITD=14NT=60NI=20.rr");
       
        ArrayList<String> lines=new ArrayList<>();
        ArrayList<Double> JSlist=new ArrayList<>();
         try (BufferedReader bufRdr1 = new BufferedReader(new FileReader(input)))
        {
            String line;
            String label="";
            int count=0, useJSPval=1;
            while ((line = bufRdr1.readLine()) != null)
            {
                
                line = line.trim();
                String tmp[]=line.split("\t");
                if(line.contains("W1R:") || line.contains("W2R:")){
                    count++;
                    String tmpS=line.substring(5,line.length());
                    if(count==1)
                    label+=tmpS+"\t\n";
                    else
                        label+=tmpS+"\t\n";
                }
                
                if(count==2 && useJSPval==0){
                    lines.add(label);
                    count=0;
                    label="";
                }
                
                if(useJSPval==1){
                    if(line.contains("JS: ")){
                        JSlist.add(Double.parseDouble(line.substring(4,line.length())));
                        String tmpS=String.format(Locale.ENGLISH,"%.2g", Double.parseDouble(line.substring(4,line.length()))); 
                        System.out.println("tmpS: "+tmpS);
                        label+="JS: "+tmpS+", ";
                        count++;
                    }
                    
                    if(line.contains("p-value :")){
                        String tmpS=String.format(Locale.ENGLISH,"%.2g", Double.parseDouble(line.substring(9,line.length()))); 
                        System.out.println("tmpS pval: "+tmpS);
                        label+="p-value: "+tmpS+"\t\n";
                        count++;
                    }
                    
                }
                
                if(count==4 && useJSPval==1){
                    lines.add(label);
                    count=0;
                    label="";
                }
            }
            bufRdr1.close();
        }
       catch(Exception e){
           e.printStackTrace();
       }
         
         
         File output=new File("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\Redescriptions+distributionCP\\RowLabels.txt");
         
         try
            {
                FileWriter fw = new FileWriter(output);
                
                for(int j=0;j<lines.size();j++){
                    if(j+1<lines.size())
                        fw.write(lines.get(j));
                    else
                        fw.write(lines.get(j));
                }
                   
                fw.close();
           
            }
           catch(Exception e){
                e.printStackTrace();
            }
         
         
          output=new File("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\Redescriptions+distributionCP\\JS.txt");
         
         try
            {
                FileWriter fw = new FileWriter(output);
                
                for(int j=0;j<JSlist.size();j++){
                    if(j+1<JSlist.size())
                        fw.write(JSlist.get(j)+"\n");
                    else
                        fw.write(JSlist.get(j)+"");
                }
                   
                fw.close();
           
            }
           catch(Exception e){
                e.printStackTrace();
            }
        
    }
}
