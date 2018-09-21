/*
 *  BSD License (http://lemurproject.org/galago-license)
 */
package org.lemurproject.galago.core.retrieval.prf;

import org.lemurproject.galago.core.index.stats.FieldStatistics;
import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.query.Node;
import org.lemurproject.galago.core.retrieval.query.NodeParameters;
import org.lemurproject.galago.core.retrieval.query.StructuredQuery;
import org.lemurproject.galago.core.util.TextPartAssigner;
import org.lemurproject.galago.utility.Parameters;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 *
 * Q-M model from  "Query Expansion Using Word Embeddings" (S.Kuzi et al. 2016).
 *
 * @author fedorn
 */
public class QDashMLMModel implements ExpansionModel {

  private static final Logger logger = Logger.getLogger("QDashMLM");
  private double defaultFbOrigWeight; // lambda in paper
  private int defaultNu; // how much terms to take for expansion
  private Map<String, List<WeightedTerm>> weighedTermsByQuery = new HashMap<>();
  public static final String UNIGRAM_FIELD_PREFIX = "uni-";
  protected final List<String> fields;
  protected final Parameters fieldWeights;
  protected final Retrieval retrieval;
  protected final FieldStats fieldStats;
  protected final Parameters globals;


  public QDashMLMModel(Retrieval r) throws Exception {
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

    this.retrieval = r;
    this.globals = r.getGlobalParameters();
    if (globals.isList("fields", String.class)) {
      this.fields = (List<String>) globals.getAsList("fields");
    } else {
      throw new IllegalArgumentException("MLMTraversal requires having 'fields' parameter initialized");
    }
    fieldStats = new FieldStats();
    this.fieldWeights = globals.isMap("fieldWeights") ? globals.getMap("fieldWeights") : null;

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
    Node expNode = generateExpansionQuery(weighedTerms, queryParameters);
    Node rm3 = new Node("combine");
    rm3.addChild(root);
    rm3.addChild(expNode);
    rm3.getNodeParameters().set("0", fbOrigWeight);
    rm3.getNodeParameters().set("1", 1.0 - fbOrigWeight);
    return rm3;
  }

  public Node generateExpansionQuery(List<WeightedTerm> weightedTerms, Parameters queryParameters) throws Exception {
    Node expNode = new Node("combine");
    for (int i = 0; i < weightedTerms.size(); i++) {
      Node expChild = getUnigramNode(queryParameters, weightedTerms.get(i).getTerm());
      expNode.addChild(expChild);
      expNode.getNodeParameters().set("" + i, weightedTerms.get(i).getWeight());
    }
    return expNode;
  }

  protected Node getUnigramNode(Parameters queryParameters, String term) throws Exception {
    String scorerType = queryParameters.get("scorer", globals.get("scorer", "dirichlet"));
    String fieldCombineOperator = queryParameters.get("fcombop", globals.get("fcombop", "wsum"));

    ArrayList<Node> termFields = new ArrayList<Node>();
    NodeParameters nodeweights = new NodeParameters();
    Parameters availableParts = this.retrieval.getAvailableParts();
    int i = 0;
    double normalizer = 0.0;
    for (String field : fields) {
      Node termFieldCounts, termExtents;

      // if we have access to the correct field-part:
      if (availableParts.containsKey("field.krovetz." + field) ||
              availableParts.containsKey("field.porter." + field) ||
              availableParts.containsKey("field." + field)) {
        NodeParameters par1 = new NodeParameters();
        par1.set("default", term);
        termFieldCounts = TextPartAssigner.assignFieldPart(new Node("counts", par1, new ArrayList()), availableParts, field);
      } else {
        // otherwise use an #inside op
        NodeParameters par1 = new NodeParameters();
        par1.set("default", term);
        termExtents = new Node("extents", par1, new ArrayList());
        termExtents = TextPartAssigner.assignPart(termExtents, globals, availableParts);

        termFieldCounts = new Node("inside");
        termFieldCounts.addChild(StructuredQuery.parse("#extents:part=extents:" + field + "()"));
        termFieldCounts.addChild(termExtents);
      }

      double fieldWeight = 0.0;
      if (fieldWeights != null && fieldWeights.containsKey(UNIGRAM_FIELD_PREFIX + field)) {
        fieldWeight = fieldWeights.getDouble(UNIGRAM_FIELD_PREFIX + field);
      } else {
        //fieldWeight = queryParameters.get(UNIGRAM_FIELD_PREFIX + field, 0.0);
        fieldWeight = queryParameters.get(UNIGRAM_FIELD_PREFIX + field, 1.0);
      }
      nodeweights.set(Integer.toString(i), fieldWeight);
      normalizer += fieldWeight;

      Node termScore = new Node(scorerType);
      termScore.getNodeParameters().set("lengths", field);
      if (scorerType.equals("dirichlet")) {
        if (queryParameters.containsKey("mu-" + field))
          termScore.getNodeParameters().set("mu", queryParameters.getDouble("mu-" + field));
        else if (globals.containsKey("mu-" + field))
          termScore.getNodeParameters().set("mu", globals.getDouble("mu-" + field));
      }
      if (scorerType.equals("bm25")) {
        if (queryParameters.containsKey("smoothing_" + field))
          termScore.getNodeParameters().set("b", queryParameters.getDouble("smoothing_" + field));
        else if (globals.containsKey("smoothing_" + field))
          termScore.getNodeParameters().set("b", globals.getDouble("smoothing_" + field));
      }
      termScore.addChild(fieldStats.fieldLenNodes.get(field).clone());
      termScore.addChild(termFieldCounts);
      termFields.add(termScore);
      i++;
    }
    // normalize field weights
    if (normalizer != 0) {
      for (i = 0; i < fields.size(); i++) {
        String key = Integer.toString(i);
        nodeweights.set(key, nodeweights.getDouble(key) / normalizer);
      }
    }

    return new Node(fieldCombineOperator, nodeweights, termFields);
  }

  protected class FieldStats {
    private final Map<String, FieldStatistics> fieldStats = new HashMap();
    private final Map<String, Node> fieldLenNodes = new HashMap();

    FieldStats() {
      if (fields == null) throw new IllegalStateException("Fields must be initialized");
      if (retrieval == null) throw new IllegalStateException("Retrieval must be initialized");
      try {
        for (String field : fields) {
          Node fieldLen = StructuredQuery.parse("#lengths:" + field + ":part=lengths()");
          FieldStatistics fieldStat = retrieval.getCollectionStatistics(fieldLen);
          fieldStats.put(field, fieldStat);
          fieldLenNodes.put(field, fieldLen);
        }
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    }

    protected Map<String, Node> getFieldLenNodes() {
      return fieldLenNodes;
    }

    protected Map<String, FieldStatistics> getFieldStats() {
      return fieldStats;
    }
  }
}
