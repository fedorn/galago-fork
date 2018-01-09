// BSD License (http://www.galagosearch.org/license)
package org.lemurproject.galago.contrib.retrieval.traversal;

import org.lemurproject.galago.core.index.stats.FieldStatistics;
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

/**
 * Original BM25F from "Simple BM25 Extension to Miltiple Weighted Fields"
 *
 * @author fedorn
 */

@ImplementsOperator("origbm25f")
public class OrigBM25FTraversal extends Traversal {

  List<String> fieldList;
  Parameters availableFields, weights;
  Retrieval retrieval;

  public OrigBM25FTraversal(Retrieval retrieval) {
    this.retrieval = retrieval;
    Parameters globals = retrieval.getGlobalParameters();
    weights = globals.containsKey("bm25f") ? globals.getMap("bm25f") : Parameters.create();
    fieldList = globals.getAsList("fields", String.class);
    try {
      availableFields = retrieval.getAvailableParts();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  public void beforeNode(Node original, Parameters queryParams) throws Exception {
  }

  public Node afterNode(Node original, Parameters queryParams) throws Exception {
    if (original.getOperator().equals("origbm25f")) {
      // Create the replacing root
      NodeParameters rootP = new NodeParameters();
      Parameters cumulativeWeights = weights.get("weights", Parameters.create());
      Node newRoot = new Node("origbm25fcomb", rootP);
      newRoot.getNodeParameters().set("norm", false);
      // Now generate the field-based subtrees for all extent/count nodes
      // NOTE : THIS IS BROKEN. IT WON'T RECOGNIZE WINDOW COUNT NODES, BUT IT SHOULD
      List<Node> children = original.getInternalNodes();
      queryParams.set("numPotentials", children.size());
      for (int i = 0; i < children.size(); i++) {
        Node termNode = children.get(i);
        double idf = getIDF(termNode);
        Node termCombiner = createFieldsOfTerm(termNode, weights.get("b", queryParams.get("b", 0.5)), cumulativeWeights, i, weights.get("K", queryParams.get("K", 0.5)),
                idf, queryParams);
        newRoot.addChild(termCombiner);
        newRoot.getNodeParameters().set("idf" + i, idf);
      }
      return newRoot;
    } else {
      return original;
    }
  }

  private double getIDF(Node termNode) throws Exception {
    // get the global document count:
    FieldStatistics cs = retrieval.getCollectionStatistics("#lengths:part=lengths()");
    double documentCount = cs.documentCount;

    // get the number of documents this term occurs in:
    termNode.getNodeParameters().set("queryType", "count");
    termNode = retrieval.transformQuery(termNode, Parameters.create());
    NodeStatistics ns = retrieval.getNodeStatistics(termNode.toString());
    long df = ns.nodeDocumentCount;

    // compute idf and return
    return Math.log(documentCount / (df + 0.5));
  }

  private Node createFieldsOfTerm(Node termNode, double b,
          Parameters cumulativeWeights, int pos, double K, double idf, Parameters queryParams) throws Exception {
    String term = termNode.getDefaultParameter();

    // Use a straight weighting - no weight normalization
    Node combiner = new Node("origbm25ffieldcombine", new ArrayList<Node>());
    combiner.getNodeParameters().set("norm", false);
    combiner.getNodeParameters().set("K", K);
    combiner.getNodeParameters().set("b", b);

    for (String field : fieldList) {
      // Actual count node
      NodeParameters np = new NodeParameters();
      np.set("default", term);
      Node fieldTermNode = new Node("extents", np);
      fieldTermNode = TextPartAssigner.assignFieldPart(fieldTermNode, availableFields, field);

      // Now wrap it in the scorer
      np = new NodeParameters();
      np.set("lengths", field);
      np.set("pIdx", pos);
      np.set("w", cumulativeWeights.get(field, queryParams.get("weight_" + field, weights.get("weight_default", 0.5))));
      Node fieldScoreNode = new Node("origbm25field", np);
      fieldScoreNode.addChild(fieldTermNode);
      combiner.getNodeParameters().set(Integer.toString(combiner.getInternalNodes().size()),
              cumulativeWeights.get(field, queryParams.get("weight_" + field, weights.get("weight_default", 0.5))));
      combiner.addChild(fieldScoreNode);
    }

    Node lengthScorerNode = new Node("origbm25flength");
    lengthScorerNode.addChild(termNode);
    combiner.addChild(lengthScorerNode);

    return combiner;
  }
}
