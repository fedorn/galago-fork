/*
 *  BSD License (http://lemurproject.org/galago-license)
 */
package org.lemurproject.galago.core.retrieval.prf;

import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.query.Node;
import org.lemurproject.galago.core.util.WordLists;
import org.lemurproject.galago.utility.Parameters;

import javax.sound.midi.SysexMessage;
import java.io.*;
import java.util.*;
import java.util.logging.Logger;

/**
 *
 * Q-M model from  "Query Expansion Using Word Embeddings" (S.Kuzi et al. 2016).
 *
 * @author fedorn
 */
public class QDashModel implements ExpansionModel {

  private static final Logger logger = Logger.getLogger("QDash");
  private double defaultFbOrigWeight; // lambda in paper
  private int defaultNu; // how much terms to take for expansion
  private Map<String, List<WeightedTerm>> weighedTermsByQuery = new HashMap<>();


  public QDashModel(Retrieval r) throws Exception {
    defaultFbOrigWeight = r.getGlobalParameters().get("fbOrigWeight", 0.25);
    defaultNu = r.getGlobalParameters().get("nu", 25);

    Parameters weighedTermsByQueryParams = r.getGlobalParameters().getMap("qmtermscores");

    for (String queryId : weighedTermsByQueryParams.keySet()) {
      List<WeightedTerm> queryWeightedTerms = new ArrayList<>();
      for (List termPair : weighedTermsByQueryParams.getList(queryId, List.class)) {
        String term = (String) termPair.get(0);
        double score = (double) termPair.get(1);
        queryWeightedTerms.add(new RelevanceModel1.WeightedUnigram(term, score));
      }
      weighedTermsByQuery.put(queryId, queryWeightedTerms);
    }
  }

  @Override
  public Node expand(Node root, Parameters queryParameters) throws Exception {

    double fbOrigWeight = queryParameters.get("fbOrigWeight", defaultFbOrigWeight);
    int nu = queryParameters.get("nu", defaultNu);
    if (fbOrigWeight == 1.0) {
      logger.info("fbOrigWeight is invalid (1.0)");
      return root;
    }
    if (!weighedTermsByQuery.containsKey(queryParameters.getString("number"))) {
      logger.info("No expansion terms for query " + queryParameters.getString("number"));
      return root;
    }
    List<WeightedTerm> weighedTerms = weighedTermsByQuery.get(queryParameters.getString("number")).subList(0, nu);
    Node expNode = generateExpansionQuery(weighedTerms);
    Node rm3 = new Node("combine");
    rm3.addChild(root);
    rm3.addChild(expNode);
    rm3.getNodeParameters().set("0", fbOrigWeight);
    rm3.getNodeParameters().set("1", 1.0 - fbOrigWeight);
    return rm3;
  }

  public Node generateExpansionQuery(List<WeightedTerm> weightedTerms) throws IOException {
    Node expNode = new Node("combine");
    for (int i = 0; i < weightedTerms.size(); i++) {
      Node expChild = new Node("text", weightedTerms.get(i).getTerm());
      expNode.addChild(expChild);
      expNode.getNodeParameters().set("" + i, weightedTerms.get(i).getWeight());
    }
    return expNode;
  }
}
