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
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.commons.math3.distribution.BinomialDistribution;
import static redescriptionmining.SettingsReader.ENCODING;
/**
 *
 * @author matej
 */
public class LoadRedescriptions {

        public static void main(String[] args) {
            
            
        ApplicationSettings appset=new ApplicationSettings();
        appset.readSettings(new File("C:\\Users\\matej\\Documents\\Redescription mining with CLUS\\Settings.set"));
        System.out.println("Num targets: "+appset.numTargets);
        System.out.println("Num trees in RS: "+appset.numTreesinForest);
        System.out.println("Average tree depth in RS: "+appset.aTreeDepth);
        System.out.println("Allow left side rule negation: "+appset.leftNegation);
        System.out.println("Allow right side rule negation: "+appset.rightNegation);
        System.out.println("Allow left side rule disjunction: "+appset.leftDisjunction);
        System.out.println("Allow right side rule disjunction: "+appset.rightDisjunction);
        
        Mappings fid=new Mappings();
        DataSetCreator dat=new DataSetCreator(appset.viewInputPaths.get(0)/*appset.outFolderPath+"\\input1.arff"*/,appset.viewInputPaths.get(1)/*appset.outFolderPath+"\\input2.arff"*/ , appset.outFolderPath);
        fid.createIndex(appset.outFolderPath+"\\Jinput.arff");
            
            File input=new File("C:\\Users\\matej\\Downloads\\python-siren-3.0.0 (3) - Copy\\tmp.txt");
            BufferedReader reader;
            
            String dataInput="";//add path
            //DataSetCreator dat=new DataSetCreator(dataInput);
           // ArrayList<RedescriptionReReMi> red=new ArrayList<>();
            RedescriptionSet rs=new RedescriptionSet();
        String file="";
         
        try {
      Path path =Paths.get(input.getAbsolutePath());
      System.out.println("Path: "+input.getAbsolutePath());
      reader = Files.newBufferedReader(path,ENCODING);
      String line = null;
      int count=0;
     
      Mappings map=new Mappings();
      
      int ElemCount=0;
      int AttrCount=0;
      
      String regex="v[0-9]+";
      Pattern p=Pattern.compile(regex);
      
        System.out.println("Number of attributes: "+(dat.schema.getNbAttributes()-1));
        System.out.println("W2 index: "+dat.W2indexs.get(0));
      
      //RedescriptionReReMi r=null;
       Redescription r=null;
      while ((line = reader.readLine()) != null){
        if(count==0){
            //r=new RedescriptionReReMi();
             r=new Redescription(dat);
      Conjunction cleft=new Conjunction();
      Conjunction cright=new Conjunction();
            String elements[]=line.split("\t");
            String elemLeft[]=elements[0].split(" ");
            String elemRight[]=elements[1].split(" ");
           
            for(int i=0;i<elemLeft.length;i++){
                if(!map.exampleId.containsKey(elemLeft[i])){
                    map.exampleId.put(elemLeft[i], ElemCount++);
                    map.idExample.put(ElemCount-1, elemLeft[i]);
                }
               // r.suppLeft.add(map.exampleId.get(elemLeft[i]));
                //cleft.elements.add(map.exampleId.get(elemLeft[i]));
                cleft.elements.add(Integer.parseInt(elemLeft[i]));
            }
            
            for(int i=0;i<elemRight.length;i++){
                if(!map.exampleId.containsKey(elemRight[i])){
                    map.exampleId.put(elemRight[i], ElemCount++);
                    map.idExample.put(ElemCount-1, elemRight[i]);
                }
                //r.suppRight.add(map.exampleId.get(elemRight[i]));
                //cright.elements.add(map.exampleId.get(elemRight[i]));
                cright.elements.add(Integer.parseInt(elemRight[i]));
            }
            count++;
            r.viewElementsLists.get(0).add(cleft);//generalize
            r.viewElementsLists.get(1).add(cright);
            TIntIterator iterator=r.viewElementsLists.get(0).get(0).elements.iterator();
            while(iterator.hasNext()){
                int elem=iterator.next();
                if(r.viewElementsLists.get(1).get(0).elements.contains(elem))
                    r.elements.add(elem);
            }
            /*for(int elem:r.leftRuleElements.get(0).elements)
                if(r.suppRight.contains(elem))
                    r.elements.add(elem);*/
            
            r.JS=(double)r.elements.size()/(r.viewElementsLists.get(0).get(0).elements.size()+r.viewElementsLists.get(1).get(0).elements.size()-r.elements.size());
            double prob=((double)(r.viewElementsLists.get(0).get(0).elements.size()*r.viewElementsLists.get(1).get(0).elements.size()))/(dat.numExamples*dat.numExamples);
            BinomialDistribution dist=new BinomialDistribution(dat.numExamples,prob);
            r.pVal=1.0-dist.cumulativeProbability(r.elements.size());
        }
        else if(count==1){
             Matcher matcher=p.matcher(line);
            System.out.println("Loading attributes left");
            while(matcher.find()){
                String attr=matcher.group();
                System.out.println("attr: "+attr);
                if(!map.attId.containsKey(attr)){
                    map.attId.put(attr, AttrCount++);
                    map.idAtt.put(AttrCount-1, attr);
                }
                attr=attr.replaceAll("v", "");
                //r.leftRuleElements.get(0).attributes.add(map.attId.get(attr));
                if(!r.viewElementsLists.get(0).get(0).attributes.contains(Integer.parseInt(attr)))
                           r.viewElementsLists.get(0).get(0).attributes.add(Integer.parseInt(attr));
                else{
                    int inserted=0;
                    for(int i=1;i<r.viewElementsLists.get(0).size();i++)
                        if(!r.viewElementsLists.get(0).get(i).attributes.contains(Integer.parseInt(attr))){
                             r.viewElementsLists.get(0).get(i).attributes.add(Integer.parseInt(attr));
                             inserted=1;
                             break;
                        }
                  
                    if(inserted==0){
                    Conjunction cleft=new Conjunction();
                    cleft.attributes.add(Integer.parseInt(attr));
                    r.viewElementsLists.get(0).add(cleft);
                    }
                    //r.viewElementsLists.get(0).add(at);
                }   
            }           
            count++;    
        }
        else if(count==2){
            Matcher matcher=p.matcher(line);
            System.out.println("Loading attributes right");
             while(matcher.find()){
                String attr=matcher.group();
                System.out.println("attr: "+attr);
                if(!map.attId.containsKey(attr)){
                    map.attId.put(attr, AttrCount++);
                    map.idAtt.put(AttrCount-1, attr);
                }
                attr=attr.replaceAll("v", "");
                //r.rightRuleElements.get(0).attributes.add(map.attId.get(attr));
              if(!r.viewElementsLists.get(1).get(0).attributes.contains(dat.W2indexs.get(0)-2+Integer.parseInt(attr)))
                r.viewElementsLists.get(1).get(0).attributes.add(dat.W2indexs.get(0)-2+Integer.parseInt(attr));
              else{
                    int inserted=0;
                    for(int i=1;i<r.viewElementsLists.get(1).size();i++)
                        if(!r.viewElementsLists.get(1).get(i).attributes.contains(dat.W2indexs.get(0)-2+Integer.parseInt(attr))){
                             r.viewElementsLists.get(1).get(i).attributes.add(dat.W2indexs.get(0)-2+Integer.parseInt(attr));
                             inserted=1;
                             break;
                        }
                  
                    if(inserted==0){
                    Conjunction cright=new Conjunction();
                    cright.attributes.add(Integer.parseInt(attr));
                    r.viewElementsLists.get(1).add(cright);
                    }
                    //r.viewElementsLists.get(0).add(at);
                }   
            }
            
            count=0;
            //red.add(r);
            rs.redescriptions.add(r);
        }
    }
      reader.close();
         }
         catch(Exception e){
             e.printStackTrace();
         }
        
        CoocurenceMatrix coc=new CoocurenceMatrix(dat.numExamples,dat.schema.getNbAttributes()-1);
         coc.computeMatrix(rs, dat);
        File out=new File(appset.outFolderPath+"\\ElementsDBLPReReMi.txt");
         coc.writeToFileElements(out, dat.numExamples);
         out=new File(appset.outFolderPath+"\\AttributesDBLPReReMi.txt");
         coc.writeToFileAttributes(out,dat.schema.getNbAttributes()-1);
         
        RedescriptionSet Result=new RedescriptionSet();
        double weights[]= {0.2,0.2,0.2,0.2,0.2};
        Result.createRedescriptionSetCooc(rs, weights, appset, dat, fid,coc);
         int numFullRed=0;
         for(int i=0;i<Result.redescriptions.size();i++){
             System.out.println("Intersection size: "+rs.redescriptions.get(i).elements.size());
             System.out.println("Attribute size: "+(rs.redescriptions.get(i).viewElementsLists.get(0).get(0).attributes.size()+rs.redescriptions.get(i).viewElementsLists.get(1).get(0).attributes.size()));
             System.out.println("JS: "+rs.redescriptions.get(i).JS);
             if(Result.redescriptions.get(i).JS==1.0){
                 numFullRed++;
             }
         }
         
      coc.init(dat.numExamples, dat.schema.getNbAttributes()-1);
      coc.computeMatrix(Result, dat);
      out=new File(appset.outFolderPath+"\\AttributesOptDBLPReReMi.txt");
      coc.writeToFileAttributes(out, dat.schema.getNbAttributes()-1);
      out=new File(appset.outFolderPath+"\\ElementsOptDBLPReReMi.txt");
      coc.writeToFileElements(out, dat.numExamples);
         
        // double weights[]={0.2,0.2,0.2,0.2,0.2};
         double coverage[]=new double[2];
         double score=Result.computeRedescriptionSetScore(weights, coverage, dat, fid);
         boolean b[]={false};
         System.out.println("Redescription set score: "+score);
         
     // Result.writeToFile(appset.outFolderPath+"\\"+"ReReMi"+appset.outputName, dat, fid, 0,numFullRed,appset, score, coverage,b);
      Result.writePlots(appset.outFolderPath+"\\"+"RuleDataCountryReReMi.csv", appset,dat,fid);
      System.out.println("Elements coverage: "+coverage[0]);
      System.out.println("Attributes coverage: "+coverage[1]);
        }      
}
