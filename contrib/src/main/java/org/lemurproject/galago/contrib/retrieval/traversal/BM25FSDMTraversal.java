// BSD License (http://www.galagosearch.org/license)
package org.lemurproject.galago.contrib.retrieval.traversal;

import org.apache.commons.lang.math.NumberUtils;
import org.lemurproject.galago.core.index.stats.FieldStatistics;
import org.lemurproject.galago.core.index.stats.NodeStatistics;
import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.ann.ImplementsOperator;
import org.lemurproject.galago.core.retrieval.query.Node;
import org.lemurproject.galago.core.retrieval.query.NodeParameters;
import org.lemurproject.galago.core.retrieval.query.StructuredQuery;
import org.lemurproject.galago.core.retrieval.traversal.Traversal;
import org.lemurproject.galago.core.util.TextPartAssigner;
import org.lemurproject.galago.utility.Parameters;

import java.util.ArrayList;
import java.util.List;

/**
 * Transforms a #bm25f operator into a full expansion of the BM25F model. That
 * means:
 *
 * Given `meg ryan war`, the output should be like:
 *
 * #bm25fcomb:K=0.5( #combine:0=0.407:1=0.382:2=0.187 (
 * #feature:bm25f:b=0.1(meg.cast) #feature:bm25f:b=0.2(meg.team)
 * #feature:bm25f:b=0.3(meg.title) ) #idf(meg) #combine:0=0.407:1=0.382:2=0.187
 * ( #feature:bm25f:b=0.1(ryan.cast) #feature:bm25f:b=0.2(ryan.team)
 * #feature:bm25f:b=0.3(ryan.title) ) #idf(ryan)
 * #combine:0=0.407:1=0.382:2=0.187 ( #feature:bm25f:b=0.1(war.cast)
 * #feature:bm25f:b=0.2(war.team) #feature:bm25f:b=0.3(war.title) ) #idf(war) )
 *
 * Except the weights involved should be tuned and not made up. Weights are:
 *
 * - b_f : A 'b' that is tuned for that field. It is NOT dependent to a query -
 * only to the field. (attached to feature nodes) - W_f : A weight for each
 * field, that is multiplied against every term for that field, making the total
 * term weight. - K : A field/query independent tunable parameter. Used for
 * smoothing term scores.
 *
 *
 * @author irmarc
 */
@ImplementsOperator("bm25f")
public class BM25FSDMTraversal extends Traversal {

  List<String> fieldList;
  Parameters availableFields, weights;
  Retrieval retrieval;

  public static final String UNIGRAM_FIELD_PREFIX = "uni-";
  public static final String ORDERED_FIELD_PREFIX = "od-";
  public static final String UNWINDOW_FIELD_PREFIX = "uww-";

  private final int windowLimitDefault;
  private final double unigramDefault;
  private final double orderedDefault;
  private final double unorderedDefault;

  private final String odOp;
  private final int odWidth;
  private final String uwOp;
  private final int uwWidth;


  public BM25FSDMTraversal(Retrieval retrieval) {
    this.retrieval = retrieval;
    Parameters globals = retrieval.getGlobalParameters();
    weights = globals.containsKey("bm25f") ? globals.getMap("bm25f") : Parameters.create();
    fieldList = globals.getAsList("fields", String.class);
    unigramDefault = globals.get("uniw", 0.8);
    orderedDefault = globals.get("odw", 0.1);
    unorderedDefault = globals.get("uww", 0.1);
    windowLimitDefault = (int) globals.get("windowLimit", 2);

    odOp = globals.get("sdm.od.op", "ordered");
    odWidth = (int) globals.get("sdm.od.width", 1);

    uwOp = globals.get("sdm.uw.op", "unordered");
    uwWidth = (int) globals.get("sdm.uw.width", 8);

    try {
      availableFields = retrieval.getAvailableParts();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  public void beforeNode(Node original, Parameters queryParams) throws Exception {
  }

  public Node afterNode(Node original, Parameters queryParams) throws Exception {
    if (original.getOperator().equals("bm25fsdm")) {
      // Create the replacing root
      NodeParameters rootP = new NodeParameters();
      rootP.set("K", weights.get("K", queryParams.get("K", 0.5)));
      Parameters cumulativeWeights = weights.get("weights", Parameters.create());
      Parameters smoothing = weights.get("smoothing", Parameters.create());
      Node uniRoot = new Node("bm25fcomb", rootP);
      uniRoot.getNodeParameters().set("norm", false);
      // Now generate the field-based subtrees for all extent/count nodes
      // NOTE : THIS IS BROKEN. IT WON'T RECOGNIZE WINDOW COUNT NODES, BUT IT SHOULD
      List<Node> children = original.getInternalNodes();
      queryParams.set("numPotentials", children.size());
      for (int i = 0; i < children.size(); i++) {
        Node termNode = children.get(i);
        double idf = getIDF(termNode);
        Node termCombiner = createFieldsOfTerm(termNode, smoothing, cumulativeWeights, i, weights.get("K", queryParams.get("K", 0.5)),
                idf, queryParams);
        uniRoot.addChild(termCombiner);
        uniRoot.getNodeParameters().set("idf" + i, idf);
      }

      if (children.size() == 1) {
        return uniRoot;
      }

      int windowLimit = (int) queryParams.get("windowLimit", windowLimitDefault);

      rootP = new NodeParameters();
      rootP.set("K", weights.get("K", queryParams.get("K", 0.5)));
      Node odRoot = new Node("bm25fcomb", rootP);
      odRoot.getNodeParameters().set("norm", false);

      rootP = new NodeParameters();
      rootP.set("K", weights.get("K", queryParams.get("K", 0.5)));
      Node uwRoot = new Node("bm25fcomb", rootP);
      uwRoot.getNodeParameters().set("norm", false);

      int bigramI = 0;
      for (int n = 2; n <= windowLimit; n++) {
        for (int i = 0; i < (children.size() - n + 1); i++) {
          List<Node> seq = children.subList(i, i + n);
          String orderedOp = this.odOp;
          String unorderedOp = this.uwOp;

          Node odNode = new Node(queryParams.get("sdm.od.op", orderedOp), new NodeParameters(queryParams.get("sdm.od.width", odWidth)), Node.cloneNodeList(seq));
          Node uwwNode = new Node(queryParams.get("sdm.uw.op", unorderedOp), new NodeParameters(queryParams.get("sdm.uw.width", uwWidth)), Node.cloneNodeList(seq));

          double odIdf = getIDF(odNode);
          double uwwIdf = getIDF(uwwNode);

          Node odCombiner = createFieldsOfBigram(seq, smoothing, cumulativeWeights, i, weights.get("K", queryParams.get("K", 0.5)),
                  odIdf, queryParams, true);
          Node uwwCombiner = createFieldsOfBigram(seq, smoothing, cumulativeWeights, i, weights.get("K", queryParams.get("K", 0.5)),
                  uwwIdf, queryParams, false);


          odRoot.addChild(odCombiner);
          odRoot.getNodeParameters().set("idf" + bigramI, odIdf);
          uwRoot.addChild(uwwCombiner);
          uwRoot.getNodeParameters().set("idf" + bigramI, uwwIdf);
          ++bigramI;
        }
      }

      NodeParameters weights = new NodeParameters();
      ArrayList<Node> immediateChildren = new ArrayList<Node>();

      double unigramW = queryParams.get("uniw", unigramDefault);
      double orderedW = queryParams.get("odw", orderedDefault);
      double unorderedW = queryParams.get("uww", unorderedDefault);

      // unigrams
      weights.set("0", unigramW);
      immediateChildren.add(uniRoot);

      // ordered
      weights.set("1", orderedW);
      immediateChildren.add(odRoot);

      // unordered
      weights.set("2", unorderedW);
      immediateChildren.add(uwRoot);

      return new Node("combine", weights, immediateChildren, original.getPosition());
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

  private Node createFieldsOfTerm(Node termNode, Parameters smoothingWeights,
          Parameters cumulativeWeights, int pos, double K, double idf, Parameters queryParams) throws Exception {
    String term = termNode.getDefaultParameter();

    // Use a straight weighting - no weight normalization
    Node combiner = new Node("combine", new ArrayList<Node>());
    combiner.getNodeParameters().set("norm", false);

    for (String field : fieldList) {
      // Actual count node
      NodeParameters np = new NodeParameters();
      np.set("default", term);
      Node fieldTermNode = new Node("extents", np);
      fieldTermNode = TextPartAssigner.assignFieldPart(fieldTermNode, availableFields, field);

      // Now wrap it in the scorer
      np = new NodeParameters();
      np.set("b", smoothingWeights.get(field, queryParams.get("smoothing_" + field, weights.get("smoothing_default", 0.5))));
      np.set("lengths", field);
      np.set("pIdx", pos);
      np.set("K", K);
      np.set("idf", idf);
      np.set("w", cumulativeWeights.get(field, queryParams.get("weight_" + field, weights.get("weight_default", 0.5))));
      Node fieldScoreNode = new Node("bm25field", np);
      fieldScoreNode.addChild(fieldTermNode);
      combiner.getNodeParameters().set(Integer.toString(combiner.getInternalNodes().size()),
              cumulativeWeights.get(field, queryParams.get("weight_" + field, weights.get("weight_default", 0.5))));
      combiner.addChild(fieldScoreNode);
    }

    return combiner;
  }

  private Node createFieldsOfBigram(List<Node> seq, Parameters smoothingWeights,
                                  Parameters cumulativeWeights, int pos, double K, double idf, Parameters queryParams, boolean ordered) throws Exception {

    // Use a straight weighting - no weight normalization
    Node combiner = new Node("combine", new ArrayList<Node>());
    combiner.getNodeParameters().set("norm", false);

    for (String field : fieldList) {
      // Actual count node
      NodeParameters np = new NodeParameters();

      Node operationNode;
      if (ordered)
        operationNode = new Node(odOp, new NodeParameters(np.get("od.width", odWidth)));
      else {
        long unorderedWindow = np.get(("uw." + field + ".width"), np.get("uw.width", uwWidth));
        operationNode = new Node(uwOp, new NodeParameters(unorderedWindow));
      }


      for (Node t : seq) {
        String inFieldTerm = t.getNodeParameters().getAsSimpleString("default");
        if (NumberUtils.isNumber(inFieldTerm)) inFieldTerm = "@/" + inFieldTerm + "/";
        if (ordered)
         operationNode.addChild(TextPartAssigner.assignFieldPart(StructuredQuery.parse("#extents:" + inFieldTerm + "()"), this.retrieval.getAvailableParts(), field));
        else
         operationNode.addChild(TextPartAssigner.assignFieldPart(StructuredQuery.parse("#extents:" + inFieldTerm + "()"), this.retrieval.getAvailableParts(), field));
      }


      // Now wrap it in the scorer
      np = new NodeParameters();
      np.set("b", smoothingWeights.get(field, queryParams.get("smoothing_" + field, weights.get("smoothing_default", 0.5))));
      np.set("lengths", field);
      np.set("pIdx", pos);
      np.set("K", K);
      np.set("idf", idf);
      np.set("w", cumulativeWeights.get(field, queryParams.get("weight_" + field, weights.get("weight_default", 0.5))));
      Node fieldScoreNode = new Node("bm25field", np);
      fieldScoreNode.addChild(operationNode);
      combiner.getNodeParameters().set(Integer.toString(combiner.getInternalNodes().size()),
              cumulativeWeights.get(field, queryParams.get("weight_" + field, weights.get("weight_default", 0.5))));
      combiner.addChild(fieldScoreNode);
    }

    return combiner;
  }
}
