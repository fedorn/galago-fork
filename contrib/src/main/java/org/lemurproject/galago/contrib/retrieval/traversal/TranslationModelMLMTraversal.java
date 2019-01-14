// BSD License (http://www.galagosearch.org/license)
package org.lemurproject.galago.contrib.retrieval.traversal;

import org.lemurproject.galago.core.index.stats.FieldStatistics;
import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.ann.ImplementsOperator;
import org.lemurproject.galago.core.retrieval.query.Node;
import org.lemurproject.galago.core.retrieval.query.NodeParameters;
import org.lemurproject.galago.core.retrieval.query.StructuredQuery;
import org.lemurproject.galago.core.retrieval.traversal.Traversal;
import org.lemurproject.galago.core.util.TextPartAssigner;
import org.lemurproject.galago.utility.Parameters;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Translation Model from "Generalizing Translation Models in the Probabilistic Relevance Framework", Eqs. (19)-(21)
 *
 * @author fedorn
 */

@ImplementsOperator("tmmlm")
public class TranslationModelMLMTraversal extends Traversal {
  public static final String UNIGRAM_FIELD_PREFIX = "uni-";

  private static final Logger logger = Logger.getLogger("TranslationModel");
  Parameters allRelatedTerms;
  Retrieval retrieval;
  protected final List<String> fields;
  protected final Parameters fieldWeights;
  protected final FieldStats fieldStats;
  protected final Parameters globals;

  public TranslationModelMLMTraversal(Retrieval retrieval) {
    this.retrieval = retrieval;
    this.globals = retrieval.getGlobalParameters();
    allRelatedTerms = globals.getMap("transprobs");

    if (globals.isList("fields", String.class)) {
      this.fields = (List<String>) globals.getAsList("fields");
    } else {
      throw new IllegalArgumentException("TranslationModelMLMTraversal requires having 'fields' parameter initialized");
    }
    fieldStats = new FieldStats();
    this.fieldWeights = globals.isMap("fieldWeights") ? globals.getMap("fieldWeights") : null;
  }

  public void beforeNode(Node original, Parameters queryParams) throws Exception {
  }

  public Node afterNode(Node original, Parameters queryParams) throws Exception {
    if (original.getOperator().equals("tmmlm")) {
      Parameters availableParts = this.retrieval.getAvailableParts();
      // Create the replacing root
      Node newRoot = new Node("combine");
      List<Node> children = original.getInternalNodes();
      for (int i = 0; i < children.size(); i++) {
        Node termNode = children.get(i);
        String term = termNode.getDefaultParameter();

        if (allRelatedTerms.containsKey(term.toLowerCase())) {
          ArrayList<Node> termFields = new ArrayList<Node>();
          NodeParameters nodeweights = new NodeParameters();
          int j = 0;
          double normalizer = 0.0;
          for (String field : fields) {
            NodeParameters np = new NodeParameters();
            np.set("default", term);
            Node termCountsNode = TextPartAssigner.assignFieldPart(new Node("counts", np, new ArrayList()), availableParts, field);

            double fieldWeight = 0.0;
            if (fieldWeights != null && fieldWeights.containsKey(UNIGRAM_FIELD_PREFIX + field)) {
              fieldWeight = fieldWeights.getDouble(UNIGRAM_FIELD_PREFIX + field);
            } else {
              //fieldWeight = queryParameters.get(UNIGRAM_FIELD_PREFIX + field, 0.0);
              fieldWeight = queryParams.get(UNIGRAM_FIELD_PREFIX + field, 1.0);
            }
            nodeweights.set(Integer.toString(j), fieldWeight);
            normalizer += fieldWeight;

            double background = getBackground(field, termCountsNode);
            Node termCombiner = createExpansionOfTerm(field, termCountsNode, allRelatedTerms.getList(term.toLowerCase(), List.class),
                    background, queryParams.get("expWeight", globals.getDouble("expWeight")));
            termFields.add(termCombiner);
            j++;
          }
          // normalize field weights
          if (normalizer != 0) {
            for (j = 0; j < fields.size(); j++) {
              String key = Integer.toString(j);
              nodeweights.set(key, nodeweights.getDouble(key) / normalizer);
            }
          }
          newRoot.addChild(new Node("wsum", nodeweights, termFields));
        } else {
          logger.info("no expansion terms for: " + term);
          newRoot.addChild(getUnigramNode(queryParams, term));
        }
      }
      newRoot.getNodeParameters().set("norm", false);
      return newRoot;
    } else {
      return original;
    }
  }

  private double getBackground(String field, Node termNode) throws Exception {
    long collectionLength = fieldStats.getFieldStats().get(field).collectionLength;
    long collectionFrequency = retrieval.getNodeStatistics(termNode).nodeFrequency;
    double background = (collectionFrequency > 0)
            ? (double) collectionFrequency / (double) collectionLength
            : 0.5 / (double) collectionLength;

    return background;
  }

  private Node createExpansionOfTerm(String field, Node termNode, List<List> relatedTerms, double background, double expWeight) throws IOException {
    String term = termNode.getDefaultParameter();

    // Use a straight weighting - no weight normalization
    Node combiner = new Node("tmtermcombine", new ArrayList<Node>());
    combiner.getNodeParameters().set("background", background);
    combiner.getNodeParameters().set("norm", false); // To prevent normalization of weights, important parameter!

//    combiner.getNodeParameters().set("0", 1.0);
//    combiner.addChild(termNode);
    NodeParameters np = new NodeParameters();
    np.set("default", term);
    Node origTermNode = TextPartAssigner.assignFieldPart(new Node("counts", np, new ArrayList()), this.retrieval.getAvailableParts(), field);

    // Now wrap it in the scorer
    np = new NodeParameters();
    Node scoreNode = new Node("origbm25field", np); // not related to bm25f, simply returns count as a score
    scoreNode.getNodeParameters().set("lengths", field);
    scoreNode.addChild(origTermNode);
    combiner.getNodeParameters().set("0", 1.0);
    combiner.addChild(scoreNode);

    for (List termPair : relatedTerms.subList(0, Math.min(1000, relatedTerms.size()))) {
      String relatedTerm = (String) termPair.get(0);
      double score = (double) termPair.get(1);

      // Actual count node
      np = new NodeParameters();
      np.set("default", relatedTerm);
      Node relatedTermNode = TextPartAssigner.assignFieldPart(new Node("counts", np, new ArrayList()), this.retrieval.getAvailableParts(), field);

      // Now wrap it in the scorer
      np = new NodeParameters();
      scoreNode = new Node("origbm25field", np); // not related to bm25f, simply returns count as a score
      scoreNode.getNodeParameters().set("lengths", field);
      scoreNode.addChild(relatedTermNode);
      combiner.getNodeParameters().set(Integer.toString(combiner.getInternalNodes().size()), score * expWeight);
      combiner.addChild(scoreNode);
    }

    Node lengthScorerNode = new Node("origbm25flength"); // not related to bm25f, simply returns doc length as a score
    lengthScorerNode.getNodeParameters().set("lengths", field);
    lengthScorerNode.addChild(termNode);
    combiner.addChild(lengthScorerNode);

    return combiner;
  }

  protected Node getUnigramNode(Parameters queryParameters, String term) throws Exception {
    ArrayList<Node> termFields = new ArrayList<Node>();
    NodeParameters nodeweights = new NodeParameters();
    Parameters availableParts = this.retrieval.getAvailableParts();
    int i = 0;
    double normalizer = 0.0;
    for (String field : fields) {
      Node termFieldCounts;

      NodeParameters par1 = new NodeParameters();
      par1.set("default", term);
      termFieldCounts = TextPartAssigner.assignFieldPart(new Node("counts", par1, new ArrayList()), availableParts, field);

      double fieldWeight = 0.0;
      if (fieldWeights != null && fieldWeights.containsKey(UNIGRAM_FIELD_PREFIX + field)) {
        fieldWeight = fieldWeights.getDouble(UNIGRAM_FIELD_PREFIX + field);
      } else {
        fieldWeight = queryParameters.get(UNIGRAM_FIELD_PREFIX + field, 1.0);
      }
      nodeweights.set(Integer.toString(i), fieldWeight);
      normalizer += fieldWeight;

      Node termScore = new Node("dirichlet");
      termScore.getNodeParameters().set("lengths", field);
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

    return new Node("wsum", nodeweights, termFields);
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

    protected Map<String, FieldStatistics> getFieldStats() {
      return fieldStats;
    }
  }
}
