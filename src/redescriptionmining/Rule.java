/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package redescriptionmining;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import gnu.trove.iterator.TIntIterator;
import gnu.trove.set.hash.TIntHashSet;
import java.util.Collections;
import java.util.Set;

/**
 *
 * @author matej
 */
public class Rule {
    //HashSet<Integer> elements=null;
    int trainSuppSize = 0;
    TIntHashSet elements=null;
    HashMap<Integer,ArrayList<Double>> ruleMap=null;
     TIntHashSet elementsMissing=null;
      ArrayList<Integer> targetValueCounts = null;
    //HashSet<Integer> ruleAtts=null;
    TIntHashSet ruleAtts=null;
    String rule=null;
    int numElements=0;
   // int UsedIn=0;
    
    Rule(){
         targetValueCounts = new ArrayList<>();
       // elements=new HashSet<Integer>();
        elements=new TIntHashSet(200);
        ruleMap=new HashMap<>();
        ruleAtts=new TIntHashSet(200);
    }
    
    Rule(String _rule, Mappings map){
        targetValueCounts = new ArrayList<>();
        rule=_rule;
        //elements=new HashSet<Integer>(); 
         elements=new TIntHashSet();
        ruleMap=new HashMap();
        ruleAtts=new TIntHashSet();
        
         String rs[]=rule.split(" AND ");
       // System.out.println("Rule: "+_rule);
        for(int i=0;i<rs.length;i++){
                if(rs[i].contains(">") && !rs[i].contains("=")){//add case for categorical data contains("=")
                    String t[]=rs[i].split(" > ");
                    //if(!ruleMap.containsKey(map.attId.get(t[0]))){
                     if(!ruleAtts.contains(map.attId.get(t[0]))){
                       /* ArrayList<Double> tmp=new ArrayList<Double>(Collections.nCopies(4, 0.0));
                        t[1]=t[1].replaceAll(",", ".");
                         String num[]=t[1].split("\\.");
                        String number="";
                        if(num.length>2){
                                for(int z=0;z<num.length;z++)
                                    if(z+1<num.length)
                                        number+=num[z];
                                    else
                                        number+="."+num[z];
                                t[1]=number;
                        }
                        tmp.set(0,1.0);
                        tmp.set(1,Double.parseDouble(t[1]));
                        tmp.set(3, Double.MAX_VALUE);*/
                        //ruleMap.put(map.attId.get(t[0]), null);
                         ruleAtts.add(map.attId.get(t[0]));
                    }
                    else{
                        /*ArrayList<Double> tmp=ruleMap.get(t[0]);
                        t[1]=t[1].replaceAll(",", ".");
                         String num[]=t[1].split("\\.");
                        String number="";
                        if(num.length>2){
                                for(int z=0;z<num.length;z++)
                                    if(z+1<num.length)
                                        number+=num[z];
                                    else
                                        number+="."+num[z];
                                t[1]=number;
                        }
                        Double value=Double.parseDouble(t[1]);
                        tmp.set(0, 1.0);
                        if(value>tmp.get(1)){
                            tmp.set(1,value);
                            ruleMap.put(t[0], tmp);
                        }*/
                    }
                }
                else if(rs[i].contains(">=")){
                    String t[]=rs[i].split(" >= ");
                      if(!ruleAtts.contains(map.attId.get(t[0]))){
                         ruleAtts.add(map.attId.get(t[0]));
                    }
                    else{
                    }
                }
                else if(rs[i].contains("<=")){
                     String t[]=rs[i].split(" <= ");
                    //if(!ruleMap.containsKey(map.attId.get(t[0]))){
                     if(!ruleAtts.contains(map.attId.get(t[0]))){
                      /* ArrayList<Double> tmp=new ArrayList<Double>(Collections.nCopies(4, 0.0));
                        t[1]=t[1].replaceAll(",", ".");
                      
                        String num[]=t[1].split("\\.");
                        String number="";
                       // System.out.println("num length: "+num.length);
                        if(num.length>2){
                                for(int z=0;z<num.length;z++)
                                    if(z+1<num.length)
                                        number+=num[z];
                                    else
                                        number+="."+num[z];
                                t[1]=number;
                               // System.out.println("number: "+number);
                        }
                       // System.out.println("t[1]: "+t[1]);  
                        tmp.set(2,1.0);
                        tmp.set(3,Double.parseDouble(t[1]));*/
                        //ruleMap.put(map.attId.get(t[0]), null);
                        ruleAtts.add(map.attId.get(t[0]));
                    }
                    else{
                       /*ArrayList<Double> tmp=ruleMap.get(t[0]);
                       t[1]=t[1].replaceAll(",", ".");
                        String num[]=t[1].split("\\.");
                        String number="";
                        if(num.length>2){
                                for(int z=0;z<num.length;z++)
                                    if(z+1<num.length)
                                        number+=num[z];
                                    else
                                        number+="."+num[z];
                                t[1]=number;
                        }
                        Double value=Double.parseDouble(t[1]);
                        tmp.set(2,1.0);
                        if(value<tmp.get(3)){
                            tmp.set(3, value);
                            ruleMap.put(t[0], tmp); 
                        }*/
                    }
                }
                else if(rs[i].contains("<")){
                    String t[]=rs[i].split(" < ");
                     if(!ruleAtts.contains(map.attId.get(t[0]))){
                        ruleAtts.add(map.attId.get(t[0]));
                    }
                    else{
                    }
                }
                else if(rs[i].contains("=") && !rs[i].contains("<")){
                    String t[]=rs[i].split(" = ");
                    //if(!ruleMap.containsKey(map.attId.get(t[0]))){
                    //System.out.println("Attribute: "+t[0]);
                    //if(!map.attId.containsKey(t[0]))
                    //    System.out.println("No such attribute found!");
                    if(!ruleAtts.contains(map.attId.get(t[0]))){
                        //ArrayList<Double> tmp=new ArrayList<>();
                        //tmp.add((double)map.cattAtt.get(t[0]).getValue0().get(t[1]));
                       // ruleMap.put(map.attId.get(t[0]), tmp);
                        ruleAtts.add(map.attId.get(t[0]));
                    }
                }
                else if(rs[i].contains("in")){
                    numElements=-1;
                    return;
                }
            }      
    }
    
    
 void setClassCount(HashMap<String,Integer> targets, HashMap<Integer,Integer> targetsIndex, Mappings map){
        
        TIntIterator it = elements.iterator();
        
            while(it.hasNext()){
                int el = it.next();
                if(!map.idExample.containsKey(el))
                    continue;
                String elS = map.idExample.get(el);
                int tv = targets.get(elS);
                int tind = targetsIndex.get(tv);
                
                int tvc = targetValueCounts.size();
                
                if((tind+1)>tvc){
                    for(int i=0;i<((tind+1)-tvc);i++)
                        targetValueCounts.add(0);
                }
                
                
                targetValueCounts.set(tind, targetValueCounts.get(tind)+1);
            }
        
    }   
    
  void ConstructRule(String _rule, Mappings map){
        rule=_rule;
         elements=new TIntHashSet();
        ruleMap=new HashMap();
        ruleAtts=new TIntHashSet();
        
         String rs[]=rule.split(" AND ");
        for(int i=0;i<rs.length;i++){
                if(rs[i].contains(">") ){//add case for categorical data contains("=")
                    String t[]=rs[i].split(" > ");
                     if(!ruleAtts.contains(map.attId.get(t[0]))){
                    
                         ruleAtts.add(map.attId.get(t[0]));

                             ruleMap.put(map.attId.get(t[0]), new ArrayList<Double>(Collections.nCopies(4,0.0)));
                             
                            t[1] = t[1].replaceAll(",",".");
                             int numOccOfDot = t[1].length() - t[1].replaceAll("\\.", "").length();
                             
                             if(numOccOfDot>1){
                                 while((t[1].length() - t[1].replaceAll("\\.", "").length())>1)
                                        t[1] = t[1].replaceFirst("\\.", "");
                             }
                             
                             ruleMap.get(map.attId.get(t[0])).set(1, Double.parseDouble(t[1])+0.0000001);
                             ruleMap.get(map.attId.get(t[0])).set(3,Double.POSITIVE_INFINITY);
                    }
                    else{
                       
                    }
                }
                else if(rs[i].contains("<=")){
                     String t[]=rs[i].split(" <= ");
                     if(!ruleAtts.contains(map.attId.get(t[0]))){
                        ruleAtts.add(map.attId.get(t[0]));
                             ruleMap.put(map.attId.get(t[0]), new ArrayList<Double>(Collections.nCopies(4,0.0)));
                              ruleMap.get(map.attId.get(t[0])).set(1,Double.NEGATIVE_INFINITY);
                              
                               t[1] = t[1].replaceAll(",",".");
                             int numOccOfDot = t[1].length() - t[1].replaceAll("\\.", "").length();
                             
                             if(numOccOfDot>1){
                                 while((t[1].length() - t[1].replaceAll("\\.", "").length())>1)
                                        t[1] = t[1].replaceFirst("\\.", "");
                             }
                              
                             ruleMap.get(map.attId.get(t[0])).set(3, Double.parseDouble(t[1])+0.0000001);
                         
                    }
                    else{
                    }
                }
                else if(rs[i].contains("=") && !rs[i].contains("<")){
                    String t[]=rs[i].split(" = ");

                    if(!ruleAtts.contains(map.attId.get(t[0]))){
                        int att = map.attId.get(t[0]);
                        ruleAtts.add(att);
                String val=t[1].trim();
                ArrayList<Double> attVal=new ArrayList<>();
                attVal.add((double)map.cattAtt.get(att).getValue0().get(val));
                this.ruleMap.put(att, attVal);
                    }
                }
                else if(rs[i].contains("in")){
                    numElements=-1;
                    return;
                }
            }      
    }
  
  
  void ConstructRule1(String _rule, Mappings map){
        rule=_rule;
         elements=new TIntHashSet();
        ruleMap=new HashMap();
        ruleAtts=new TIntHashSet();
        
         String rs[]=rule.split(" AND ");
        for(int i=0;i<rs.length;i++){
                if(rs[i].contains(">") && !rs[i].contains("=")){//add case for categorical data contains("=")
                    String t[]=rs[i].split(" > ");
                     if(!ruleAtts.contains(map.attId.get(t[0]))){
                    
                         ruleAtts.add(map.attId.get(t[0]));

                             ruleMap.put(map.attId.get(t[0]), new ArrayList<Double>(Collections.nCopies(4,0.0)));
                             
                            t[1] = t[1].replaceAll(",",".");
                             int numOccOfDot = t[1].length() - t[1].replaceAll("\\.", "").length();
                             
                             if(numOccOfDot>1){
                                 while((t[1].length() - t[1].replaceAll("\\.", "").length())>1)
                                        t[1] = t[1].replaceFirst("\\.", "");
                             }
                             
                             ruleMap.get(map.attId.get(t[0])).set(1, Double.parseDouble(t[1])+0.0000001);
                             ruleMap.get(map.attId.get(t[0])).set(3,Double.POSITIVE_INFINITY);
                    }
                    else{
                       
                    }
                }
                else if(rs[i].contains(">=")){
                     String t[]=rs[i].split(" >= ");
                     if(!ruleAtts.contains(map.attId.get(t[0]))){
                    
                         ruleAtts.add(map.attId.get(t[0]));

                             ruleMap.put(map.attId.get(t[0]), new ArrayList<Double>(Collections.nCopies(4,0.0)));
                             
                            t[1] = t[1].replaceAll(",",".");
                             int numOccOfDot = t[1].length() - t[1].replaceAll("\\.", "").length();
                             
                             if(numOccOfDot>1){
                                 while((t[1].length() - t[1].replaceAll("\\.", "").length())>1)
                                        t[1] = t[1].replaceFirst("\\.", "");
                             }
                             
                             ruleMap.get(map.attId.get(t[0])).set(1, Double.parseDouble(t[1])-0.0000001);
                             ruleMap.get(map.attId.get(t[0])).set(3,Double.POSITIVE_INFINITY);
                    }
                    else{
                       
                    }
                }
                else if(rs[i].contains("<=")){
                     String t[]=rs[i].split(" <= ");
                     if(!ruleAtts.contains(map.attId.get(t[0]))){
                        ruleAtts.add(map.attId.get(t[0]));
                             ruleMap.put(map.attId.get(t[0]), new ArrayList<Double>(Collections.nCopies(4,0.0)));
                              ruleMap.get(map.attId.get(t[0])).set(1,Double.NEGATIVE_INFINITY);
                              
                               t[1] = t[1].replaceAll(",",".");
                             int numOccOfDot = t[1].length() - t[1].replaceAll("\\.", "").length();
                             
                             if(numOccOfDot>1){
                                 while((t[1].length() - t[1].replaceAll("\\.", "").length())>1)
                                        t[1] = t[1].replaceFirst("\\.", "");
                             }
                              
                             ruleMap.get(map.attId.get(t[0])).set(3, Double.parseDouble(t[1])+0.0000001);
                         
                    }
                    else{
                    }
                }
                else if(rs[i].contains("<")){//new
                     String t[]=rs[i].split(" < ");
                     if(!ruleAtts.contains(map.attId.get(t[0]))){
                        ruleAtts.add(map.attId.get(t[0]));
                             ruleMap.put(map.attId.get(t[0]), new ArrayList<Double>(Collections.nCopies(4,0.0)));
                              ruleMap.get(map.attId.get(t[0])).set(1,Double.NEGATIVE_INFINITY);
                              
                               t[1] = t[1].replaceAll(",",".");
                             int numOccOfDot = t[1].length() - t[1].replaceAll("\\.", "").length();
                             
                             if(numOccOfDot>1){
                                 while((t[1].length() - t[1].replaceAll("\\.", "").length())>1)
                                        t[1] = t[1].replaceFirst("\\.", "");
                             }
                              
                             ruleMap.get(map.attId.get(t[0])).set(3, Double.parseDouble(t[1]));
                         
                    }
                    else{
                    }
                }
                else if(rs[i].contains("=") && !rs[i].contains("<")){
                    String t[]=rs[i].split(" = ");

                    if(!ruleAtts.contains(map.attId.get(t[0]))){
                        int att = map.attId.get(t[0]);
                        ruleAtts.add(att);
                String val=t[1].trim();
                ArrayList<Double> attVal=new ArrayList<>();
                attVal.add((double)map.cattAtt.get(att).getValue0().get(val));
                this.ruleMap.put(att, attVal);
                    }
                }
                else if(rs[i].contains("in")){
                    numElements=-1;
                    return;
                }
            }      
    }
   
    void addElements(Mappings map, DataSetCreator dat){ // 0 for rule, 1 if a rule is negated
          
        int inSupport = 0;
               
         for(int i=0;i<dat.numExamples;i++){
             if(!map.idExample.containsKey(i))
                 continue;
             inSupport = 1;
                         TIntIterator it=ruleAtts.iterator();
                         while(it.hasNext()){
                             int at=it.next();
                             if(!map.catAttInd.contains(at)/*!cat.contains(map.idAtt.get(at))*/){
                                if(dat.getValue(at, i)==Double.POSITIVE_INFINITY){
                                    inSupport = 0;
                                    break;
                                }
                                else{
                                    
                                    if(dat.getValue(at, i) == Double.POSITIVE_INFINITY){
                                        inSupport = 0;
                                        break;
                                    }
                                    
                                    if(!ruleMap.containsKey(at)){
                                        System.out.println("Nije sadrzan: "+map.idAtt.get(at));
                                    }

                                    if(ruleMap.get(at).size() == 1)
                                        System.out.println("Problematic attribute: "+map.idAtt.get(at));
                                    
                                    double lowerBound = ruleMap.get(at).get(1);
                                    double upperBound = ruleMap.get(at).get(3);
                                    
                                        if(dat.getValue(at, i)<=lowerBound || dat.getValue(at, i)>upperBound){
                                            inSupport = 0;
                                            break;
                                        }
                                }
                            }
                            else{
                                  Set<String> catVal=map.cattAtt.get(at).getValue0().keySet();
                                if(!catVal.contains(dat.getValueCategorical(at, i))){
                                    inSupport = 0;
                                    break;
                                }
                         }
                    }
                          if(inSupport == 1)
                                 elements.add(i);
           }
    }
    
    void addElements1(Mappings map, Mappings map1 ,DataSetCreator dat){ // 0 for rule, 1 if a rule is negated
          elements.clear();
        int inSupport = 0;
      
         for(int i=0;i<dat.numExamples;i++){
             if(!map.idExample.containsKey(i))
                 continue;
             inSupport = 1;
                         TIntIterator it=ruleAtts.iterator();
                         while(it.hasNext()){
                             int at1=it.next();
                             String atS = map1.idAtt.get(at1);
                             int at = map.attId.get(atS);
                             if(!map.catAttInd.contains(at)/*!cat.contains(map.idAtt.get(at))*/){
                                 
                                if(dat.getValue(at, i)==Double.POSITIVE_INFINITY){
                                    inSupport = 0;
                                    break;
                                }
                                else{
                                    
                                    if(dat.getValue(at, i) == Double.POSITIVE_INFINITY){
                                        inSupport = 0;
                                        break;
                                    }
                                    
                                    if(!ruleMap.containsKey(at1)){
                                        System.out.println("Nije sadrzan: "+map.idAtt.get(at));
                                    }
                                    
                                    if(ruleMap.get(at1).size() == 1)
                                        System.out.println("Problematic attribute: "+map.idAtt.get(at));
                                    
                                    double lowerBound = ruleMap.get(at1).get(1);
                                    double upperBound = ruleMap.get(at1).get(3);
                                    
                                        if(dat.getValue(at, i)<lowerBound || dat.getValue(at, i)>upperBound){
                                            inSupport = 0;
                                            break;
                                        }
                                }
                            }
                            else{
                                // this.ruleMap.put(att, attVal);
                                  Set<String> catVal=map.cattAtt.get(at).getValue0().keySet();//not ok
                                  String val = dat.getValueCategorical(at, i);
                                  int atV = (int)(double)ruleMap.get(at1).get(0);
                                  int oV = map.cattAtt.get(at).getValue0().get(val);
                                  if(atV!=oV){
                                      inSupport = 0;
                                      break;
                                  }
                                if(!catVal.contains(dat.getValueCategorical(at, i))){
                                    inSupport = 0;
                                    break;
                                }
                         }
                    }
                          if(inSupport == 1)
                                 elements.add(i);
           }
    }
       
    int numElementsInTest(Mappings map, Mappings map1 ,DataSetCreator dat){ // 0 for rule, 1 if a rule is negated
        int inSupport = 0, count = 0;
      
         for(int i=0;i<dat.numExamples;i++){
             if(!map.idExample.containsKey(i))
                 continue;
             inSupport = 1;
                         TIntIterator it=ruleAtts.iterator();
                         while(it.hasNext()){
                             int at1=it.next();
                             String atS = map1.idAtt.get(at1);
                             if(!map.attId.containsKey(atS)){
                                 System.out.println("Missing attribute in mapping: "+at1+" "+atS);
                             }
                             int at = map.attId.get(atS);
                             if(!map.catAttInd.contains(at)/*!cat.contains(map.idAtt.get(at))*/){
                                 
                                if(dat.getValue(at, i)==Double.POSITIVE_INFINITY){
                                    inSupport = 0;
                                    break;
                                }
                                else{
                                    
                                    if(dat.getValue(at, i) == Double.POSITIVE_INFINITY){
                                        inSupport = 0;
                                        break;
                                    }
                                    
                                    if(!ruleMap.containsKey(at1)){
                                        System.out.println("Nije sadrzan: "+map.idAtt.get(at));
                                    }
                                    
                                    if(ruleMap.get(at1).size() == 1)
                                        System.out.println("Problematic attribute: "+map.idAtt.get(at));
                                    
                                    double lowerBound = ruleMap.get(at1).get(1);
                                    double upperBound = ruleMap.get(at1).get(3);
                                    
                                        if(dat.getValue(at, i)<lowerBound || dat.getValue(at, i)>upperBound){
                                            inSupport = 0;
                                            break;
                                        }
                                }
                            }
                            else{
                                // this.ruleMap.put(att, attVal);
                                  Set<String> catVal=map.cattAtt.get(at).getValue0().keySet();//not ok
                                  String val = dat.getValueCategorical(at, i);
                                  int atV = (int)(double)ruleMap.get(at1).get(0);
                                  int oV = map.cattAtt.get(at).getValue0().get(val);
                                  if(atV!=oV){
                                      inSupport = 0;
                                      break;
                                  }
                                if(!catVal.contains(dat.getValueCategorical(at, i))){
                                    inSupport = 0;
                                    break;
                                }
                         }
                    }
                          if(inSupport == 1)
                                 count++;
           }
         return count;
    }
    
    int disjoint(Rule r1){
        
        String rs1 = this.rule;
        String rs2 = r1.rule;
        
        String t1[] = rs1.split(" AND ");
        String t2[] = rs2.split(" AND ");
        
        for(int i=0;i<t1.length;i++){
            t1[i] = t1[i].replaceAll("\\(1\\)", "").trim();
            if(t1[i].contains("<="))
                 t1[i] = t1[i].split(" <= ")[0].trim();
            else if(t1[i].contains(">"))
                t1[i] = t1[i].split(" > ")[0].trim();
            else if(t1[i].contains("=") && !t1[i].contains("<") && !t1[i].contains(">"))
                t1[i] = t1[i].split(" = ")[0].trim();
        }
        
         for(int i=0;i<t2.length;i++){
            t2[i] = t2[i].replaceAll("\\(1\\)", "").trim();
             if(t2[i].contains("<="))
                 t2[i] = t2[i].split(" <= ")[0].trim();
            else if(t2[i].contains(">"))
                t2[i] = t2[i].split(" > ")[0].trim();
            else if(t2[i].contains("=") && !t2[i].contains("<") && !t2[i].contains(">"))
                t2[i] = t2[i].split(" = ")[0].trim();
         }
        
        String c1,c2;
        
        for(int i=0;i<t1.length;i++){        
            for(int j=0;j<t2.length;j++)
             if(t1[i].equals(t2[j])){
                 return 0;
             }
        }
 
        return 1;
    }
    
    void addElementsMissing(Mappings map, DataSetCreator dat, int mode){ // 0 for rule, 1 if a rule is negated
        
        elementsMissing=new TIntHashSet(100);

         for(int i=0;i<dat.numExamples;i++){
            if(!elements.contains(i)){
                         TIntIterator it=ruleAtts.iterator();
                       if(mode==0){
                         while(it.hasNext()){
                             int at=it.next();
                             if(!map.catAttInd.contains(at)/*!cat.contains(map.idAtt.get(at))*/){
                                if(dat.getValue(at, i)==Double.POSITIVE_INFINITY){
                                    elementsMissing.add(i);
                                break;
                                }
                            }
                            else{
                                  Set<String> catVal=map.cattAtt.get(at).getValue0().keySet();
                                if(!catVal.contains(dat.getValueCategorical(at, i))){
                                    elementsMissing.add(i);
                                    break;
                                }
                            }
                         }
                    }
            else if(mode==1){
                int contained=1;
                 while(it.hasNext()){
                             int at=it.next();
                             if(!map.catAttInd.contains(at)/*!cat.contains(map.idAtt.get(at))*/){
                                if(dat.getValue(at, i)!=Double.POSITIVE_INFINITY){
                                    contained=0;
                                break;
                                }
                            }
                            else{
                                  Set<String> catVal=map.cattAtt.get(at).getValue0().keySet();
                                if(catVal.contains(dat.getValueCategorical(at, i))){
                                    contained=0;
                                    break;
                                }
                            }
                         }
                 
                         if(contained==1)
                             elementsMissing.add(i);
                     }
               }
           }
    }
    
    void closeInterval(DataSetCreator dat, Mappings map){
        trainSuppSize = elements.size();
         TIntIterator it = ruleAtts.iterator();
         
         while(it.hasNext()){
             ruleMap.put(it.next(), null);
         }

         Iterator<Integer> itL=ruleMap.keySet().iterator();
         
         while(itL.hasNext()){
             double min=Double.POSITIVE_INFINITY,max=Double.NEGATIVE_INFINITY;
             int att=itL.next();
         
         if(!map.cattAtt.containsKey(att)){
             //System.out.println(map.idAtt.get(att));
              TIntIterator itT=elements.iterator();
             while(itT.hasNext()){
                 
                 int s=itT.next();
           int ik=s;
        double val=dat.getValue(att, ik);
        
                if(val<min)
                    min=val;
                if(val>max)
                    max=val;
        }
             ArrayList<Double> attVal=new ArrayList<>(Collections.nCopies(4, 0.0));//leftRule.get(att);

             attVal.set(0, 1.0);
             attVal.set(2, 1.0);
             attVal.set(1, min);
             attVal.set(3, max);
             ruleMap.put(att, attVal);
             //leftRule.put(att, attVal);
         }
         else{
           TIntIterator  itT=elements.iterator();
           
             while(itT.hasNext()){
                 //System.out.println("Iterating through elements");
              int s=itT.next();
            // for(int s:leftRuleElements){
                 int ik=s;
                 String val=dat.getValueCategorical(att, ik);
                ArrayList<Double> attVal=new ArrayList<>();
                attVal.add((double)map.cattAtt.get(att).getValue0().get(val));
               // System.out.println(attVal.get(0));
                this.ruleMap.put(att, attVal);
               // leftRule.put(att, attVal);
                break;
             }
         }
       }

        TIntHashSet RSupport=new TIntHashSet(dat.numExamples);
       
         for(int i=0;i<dat.numExamples;i++){
             //Iterator<Integer> itL=leftRule.keySet().iterator();
             itL=ruleMap.keySet().iterator();
             int contained=1;
             while(itL.hasNext()){
                 int attr=itL.next();
                 
                 ArrayList<Double> attrVal=ruleMap.get(attr);
              
               if(!map.cattAtt.containsKey(attr)){
                   double val=dat.getValue(attr, i);
                 if(val>=attrVal.get(1) && val<=attrVal.get(3))
                        continue;
                 else{
                    contained=0;
                    break;
                 }
               }
               else{
                   if(attrVal == null){
                   System.out.println("CI: at: "+map.idAtt.get(attr));
                   System.out.println("attrVal size: "+attrVal.size());
                   System.out.println("attrVal val: "+attrVal.get(0));
                   }
                   String cat=map.cattAtt.get(attr).getValue1().get((int)(double)attrVal.get(0));
                   String realCat=dat.getValueCategorical(attr, i);
                   if(cat.contentEquals(realCat))
                       continue;
                   else{
                       contained=0;
                       break;
                   }
               }
           }

             if(contained==0)
                 continue;
             else{
                 RSupport.add(i);
             }
         }  
         
         if(RSupport.size()!=elements.size()){
             System.out.println("Rule size mismatch...");
             System.out.println("CLUS rule size: "+elements.size());
             System.out.println("Validation size: "+RSupport.size());
         }
       
         it=RSupport.iterator();
                 int ok=1;
                 
        while(it.hasNext()){
            int elem=it.next();
          if(!elements.contains(elem)){
              System.out.println("Not contained element in rule: "+map.idExample.get(elem));
              ok=0;
          }
        }
        
        it=elements.iterator();

        
        while(it.hasNext()){
            int elem=it.next();
            if(!RSupport.contains(elem)){
                System.out.println("Not contained element in validation: "+map.idExample.get(elem));
                ok=0;
            }
        }
        
        if(ok==0){
            System.out.println("Rule string");
        System.out.println(this.rule);
         it=elements.iterator();
        while(it.hasNext())
            System.out.print(map.idExample.get(it.next())+" ");
        System.exit(-1);
        }
  }
    
    void clearRuleMap(){
        ruleMap.clear();
    }
    
    void addRule(String _rule){
        rule=_rule;
    }
    
    void addElement(String element, Mappings map){
        if(map.exampleId.containsKey(element))
            elements.add(map.exampleId.get(element));
        /*if(rule.contentEquals("DATE = true")){
        System.out.println("Add element");
        System.out.println(element+" "+map.exampleId.get(element));
        }*/
    }
    
    void checkElements(Mappings map, DataSetCreator dat){
        Iterator<Integer> it; 
        HashSet<Integer> toRemove=new HashSet<>();
          
        TIntIterator attrsIterator = ruleAtts.iterator();
          // for(Integer att:ruleMap.keySet()){
        while(attrsIterator.hasNext()){
            int att=attrsIterator.next();
         //  for(Integer att:ruleAtts){
               if(!map.cattAtt.containsKey(att)){
                   /*System.out.println("att id: "+att);
                   System.out.println("attribute: "+map.idAtt.get(att));
                   System.out.println("type numeric...");*/
               TIntIterator iterator = elements.iterator();
                   while (iterator.hasNext()) {
                        int elem = iterator.next();
                          double val=dat.getValue(att, elem);
                         if(val==Double.POSITIVE_INFINITY){
                             toRemove.add(elem);
                         }
                    }
                   
                   /*for(Integer elem:elements){
                   double val=dat.getValue(att, elem);
                   if(val==Double.POSITIVE_INFINITY){
                 toRemove.add(elem);
               }
           }*/
               }
               else{
                   TIntIterator iterator = elements.iterator();
                  
                    while (iterator.hasNext()) {
                        int elem = iterator.next();
                          String attrVal=dat.getValueCategorical(att, elem);
                        if(!map.cattAtt.get(att).getValue0().containsKey(attrVal))
                                toRemove.add(elem);
                        
                   /*System.out.println("att id: "+att);
                   System.out.println("attribute: "+map.idAtt.get(att));
                   System.out.println("type categorical...");
                   System.out.println("Element id: "+elem);
                   System.out.println("Element: "+map.idExample.get(elem));
                   System.out.println("Value: "+attrVal);*/
                        
                    }
                   
                   /*for(Integer elem:elements){
                        String attrVal=dat.getValueCategorical(att, elem);
                        if(!map.cattAtt.get(map.idAtt.get(att)).getValue0().containsKey(attrVal))
                                toRemove.add(elem);
                 }*/
               }
          }
            
            it=toRemove.iterator();
            
            while(it.hasNext()){
                elements.remove(it.next());
            }
        
    }
    
    Rule(Rule RNeg, DataSetCreator dat, Mappings map){
           
        elements=new TIntHashSet(dat.numExamples);
        for(int i=0;i<dat.numExamples;i++){
            if(!RNeg.elements.contains(i)){
                         TIntIterator it=RNeg.ruleAtts.iterator();
                         while(it.hasNext()){
                             int at=it.next();
                             if(!map.catAttInd.contains(at)/*!cat.contains(map.idAtt.get(at))*/){
                                if(dat.getValue(at, i)!=Double.POSITIVE_INFINITY){
                                    elements.add(i);
                                break;
                                }
                            }
                            else{
                                  Set<String> catVal=map.cattAtt.get(at).getValue0().keySet();
                                if(catVal.contains(dat.getValueCategorical(at, i))){
                                    elements.add(i);
                                    break;
                                }
                            }
                         }
                    }
           }
           
           numElements=elements.size();
           ruleAtts=new TIntHashSet(RNeg.ruleAtts);
           rule=RNeg.rule;
    }
    
    public int checkConstraints(ArrayList<ArrayList<ArrayList<String>>> importantAttributes, int view, int constType ,Mappings map){  
        
        ArrayList<ArrayList<String>> constraints = importantAttributes.get(view);
        
        if(constraints.size()==0)
            return 1;
        
        TIntIterator it = this.ruleAtts.iterator();
        
        int s1=0;
        
       if(constType==2){ 
        for(int i=0;i<constraints.size();i++){
            s1=1;
            for(int j=0;j<constraints.get(i).size();j++){
                if(constraints.get(i).get(j).equals(""))
                    continue;
                /*System.out.println("constraint: "+constraints.get(i).get(j));
               System.out.println(this.rule);*/
                if(!ruleAtts.contains(map.attId.get(constraints.get(i).get(j)))){
                    s1=0;
                    break;
                } 
            }
           // System.out.println("s1: "+s1);
            if(s1==1)
                return 1;
        }
        return 0;
       }
       else if(constType==1){
            for(int i=0;i<constraints.size();i++){
                for(int j=0;j<constraints.get(i).size();j++){
                      if(constraints.get(i).get(j).equals(""))
                                 return 1;
                    if(ruleAtts.contains(map.attId.get(constraints.get(i).get(j)))){
                        return 1;
                 } 
                }
         }
        return 0;
       }
        
        return 1;
    }
    
}
