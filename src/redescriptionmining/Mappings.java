/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package redescriptionmining;

import gnu.trove.set.hash.TIntHashSet;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Iterator;
import org.javatuples.Pair;
import static redescriptionmining.SettingsReader.ENCODING;

/**
 *
 * @author matej
 */
public class Mappings {
    public HashMap<String,Integer> attId=new HashMap<>();
    HashMap<Integer,Pair<HashMap<String,Integer>,HashMap<Integer,String>>> cattAtt=new HashMap<>();
    public HashMap<Integer,String> idExample=new HashMap<>();
    public HashMap<String,Integer> exampleId=new HashMap<>();
    public HashMap<Integer,String> idAtt=new HashMap<>();
    public TIntHashSet catAttInd=new TIntHashSet();
    
    
    public void clearMaps(){
        
        attId.clear();
        cattAtt.clear();
        idExample.clear();
        exampleId.clear();
        idAtt.clear();
        catAttInd.clear();
        
    }
    
    void createIndex(String pathStr){
   BufferedReader reader;
     int attInd=0, dataSection=0,exInd=0;
         try {
      File input=new File(pathStr);
      Path path =Paths.get(input.getAbsolutePath());
      System.out.println("Path: "+input.getAbsolutePath());
      reader = Files.newBufferedReader(path,ENCODING);
      String line = null;
      while ((line = reader.readLine()) != null) {
          if(line.contains("@ATTRIBUTE") || line.contains("@attribute")){
              String tmp[]=line.split(" +"); 
              if(tmp.length==3 && tmp[2].contentEquals("numeric")){
                  attId.put(tmp[1],attInd++);
                  idAtt.put(attInd-1, tmp[1]);
              }
              else if(tmp.length==3 && tmp[2].contains("{")){
                  attId.put(tmp[1],attInd++);
                  idAtt.put(attInd-1, tmp[1]);
                  HashMap<String,Integer> lm=new HashMap<>();
                  HashMap<Integer,String> rm=new HashMap<>();
                  String catVal=tmp[2].replaceAll("\\{", "");
                  catVal=catVal.replaceAll("\\}", "");
                  String catVals[]=catVal.split(",");
                  
                  for(int i=0;i<catVals.length;i++){
                      lm.put(catVals[i], i);
                      rm.put(i, catVals[i]);
                  }
                  Pair<HashMap<String,Integer>,HashMap<Integer,String>> tmpPr=new Pair(lm,rm);
                  cattAtt.put(attId.get(tmp[1]), tmpPr);
                  catAttInd.add(this.attId.get(tmp[1]));
              }
          }
          if(line.contains("@DATA") || line.contains("@data")){
              dataSection=1;
            continue;
          }
          
          if(dataSection==1){
              String tmp[]=line.split(",");
              exampleId.put(tmp[0], exInd++);
              idExample.put(exInd-1, tmp[0]);
          }       
          
      }
      reader.close();
         }catch(IOException ioe)
            {
              System.err.println("IOException: " + ioe.getMessage());
            }
    }
    
    void printMapping(){
    /* public HashMap<String,Integer> attId=new HashMap<>();
    HashMap<String,Pair<HashMap<String,Integer>,HashMap<Integer,String>>> cattAtt=new HashMap<>();
    public HashMap<Integer,String> idExample=new HashMap<>();
    public HashMap<String,Integer> exampleId=new HashMap<>();
    public HashMap<Integer,String> idAtt=new HashMap<>();*/
        
        Iterator<String> it=attId.keySet().iterator();
        
        System.out.println("attribute id mapping...");
        while(it.hasNext()){
            String key=it.next();
            System.out.println("attr: "+key+" index: "+attId.get(key));
        }
        System.out.println();

        Iterator<Integer> it11=cattAtt.keySet().iterator();
        //it=cattAtt.keySet().iterator();
        
        while(it11.hasNext()){
            int key=it11.next();
            System.out.println("attr: "+idAtt.get(key));
            Iterator<String> it1=cattAtt.get(key).getValue0().keySet().iterator();
            
            while(it1.hasNext()){
                String k1=it1.next();
                System.out.println("catt value: "+k1+" index: "+cattAtt.get(key).getValue0().get(k1));
            }
        }
        
        it=exampleId.keySet().iterator();
        
        while(it.hasNext()){
            String key=it.next();
            System.out.println("example: "+key+" value"+exampleId.get(key));
        }
        
    }
    
}
