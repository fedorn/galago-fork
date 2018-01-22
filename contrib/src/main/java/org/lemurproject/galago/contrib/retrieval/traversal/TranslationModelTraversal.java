// BSD License (http://www.galagosearch.org/license)
package org.lemurproject.galago.contrib.retrieval.traversal;

import org.lemurproject.galago.core.index.stats.NodeStatistics;
import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.ann.ImplementsOperator;
import org.lemurproject.galago.core.retrieval.query.Node;
import org.lemurproject.galago.core.retrieval.query.NodeParameters;
import org.lemurproject.galago.core.retrieval.traversal.Traversal;
import org.lemurproject.galago.core.util.TextPartAssigner;
import org.lemurproject.galago.utility.Parameters;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

/**
 * Translation Model from "Generalizing Translation Models in the Probabilistic Relevance Framework", Eqs. (19)-(21)
 *
 * @author fedorn
 */

@ImplementsOperator("tm")
public class TranslationModelTraversal extends Traversal {

  private static final Logger logger = Logger.getLogger("TranslationModel");
  Parameters allRelatedTerms;
  Retrieval retrieval;

  public TranslationModelTraversal(Retrieval retrieval) {
    this.retrieval = retrieval;
    Parameters globals = retrieval.getGlobalParameters();
    allRelatedTerms = globals.getMap("transprobs");
  }

  public void beforeNode(Node original, Parameters queryParams) throws Exception {
  }

  public Node afterNode(Node original, Parameters queryParams) throws Exception {
    if (original.getOperator().equals("tm")) {
      // Create the replacing root
      Node newRoot = new Node("combine");
      List<Node> children = original.getInternalNodes();
      for (int i = 0; i < children.size(); i++) {
        Node termNode = children.get(i);
        String term = termNode.getDefaultParameter();

        NodeParameters np = new NodeParameters();
        np.set("default", term);
        Node termCountsNode = new Node("counts", np);

        if (allRelatedTerms.containsKey(term.toLowerCase())) {
          double background = getBackground(termCountsNode);
          Node termCombiner = createExpansionOfTerm(termCountsNode, allRelatedTerms.getList(term.toLowerCase(), List.class),
                  background, queryParams.getDouble("expWeight"));
          newRoot.addChild(termCombiner);
        } else {
          logger.info("no expansion terms for: " + term);
          newRoot.addChild(termNode);
        }
      }
      return newRoot;
    } else {
      return original;
    }
  }

  private double getBackground(Node termNode) throws Exception {
    long collectionLength = retrieval.getCollectionStatistics(new Node("lengths")).collectionLength;
    long collectionFrequency = retrieval.getNodeStatistics(termNode).nodeFrequency;
    double background = (collectionFrequency > 0)
            ? (double) collectionFrequency / (double) collectionLength
            : 0.5 / (double) collectionLength;

    return background;
  }

  private Node createExpansionOfTerm(Node termNode, List<List> relatedTerms, double background, double expWeight) {
    String term = termNode.getDefaultParameter();

    // Use a straight weighting - no weight normalization
    Node combiner = new Node("tmtermcombine", new ArrayList<Node>());
    combiner.getNodeParameters().set("background", background);
    combiner.getNodeParameters().set("norm", false); // To prevent normalization of weights, important parameter!

//    combiner.getNodeParameters().set("0", 1.0);
//    combiner.addChild(termNode);
    NodeParameters np = new NodeParameters();
    np.set("default", term);
    Node origTermNode = new Node("counts", np);

    // Now wrap it in the scorer
    np = new NodeParameters();
    Node scoreNode = new Node("origbm25field", np); // not related to bm25f, simply returns count as a score
    scoreNode.addChild(origTermNode);
    combiner.getNodeParameters().set("0", 1.0);
    combiner.addChild(scoreNode);

    for (List termPair : relatedTerms) {
      String relatedTerm = (String) termPair.get(0);
      double score = (double) termPair.get(1);

      // Actual count node
      np = new NodeParameters();
      np.set("default", relatedTerm);
      Node relatedTermNode = new Node("counts", np);

      // Now wrap it in the scorer
      np = new NodeParameters();
      scoreNode = new Node("origbm25field", np); // not related to bm25f, simply returns count as a score
      scoreNode.addChild(relatedTermNode);
      combiner.getNodeParameters().set(Integer.toString(combiner.getInternalNodes().size()), score * expWeight);
      combiner.addChild(scoreNode);
    }

    Node lengthScorerNode = new Node("origbm25flength"); // not related to bm25f, simply returns doc length as a score
    lengthScorerNode.addChild(termNode);
    combiner.addChild(lengthScorerNode);

    return combiner;
  }
}
