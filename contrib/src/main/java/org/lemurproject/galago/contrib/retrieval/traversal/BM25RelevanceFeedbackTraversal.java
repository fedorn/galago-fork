// BSD License (http://lemurproject.org/galago-license)
package org.lemurproject.galago.contrib.retrieval.traversal;

import java.io.IOException;
import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.query.Node;
import org.lemurproject.galago.core.retrieval.traversal.Traversal;
import org.lemurproject.galago.utility.Parameters;

/**
 * We run the query as a combine on the way back up, and add in the expansion
 * terms. This is similar to the RelevanceModelTraversal.
 *
 * Little weird here - we transform an operator over a subtree into low-level
 * feature operators that act on count iterators.
 *
 * @author irmarc
 */
public class BM25RelevanceFeedbackTraversal extends Traversal {
  
  Parameters globalParameters;
  Parameters availableParts;
  Retrieval retrieval;
  
  public BM25RelevanceFeedbackTraversal(Retrieval retrieval) throws IOException {
    this.retrieval = retrieval;
    this.globalParameters = retrieval.getGlobalParameters();
    this.availableParts = retrieval.getAvailableParts();
  }
  
  public Node afterNode(Node original, Parameters qp) throws Exception {
    if (original.getOperator().equals("bm25rf") == false) {
      return original;
    }

    return original;
    
//    // Kick off the inner query
//    NodeParameters parameters = original.getNodeParameters();
//    int fbDocs = (int) parameters.get("fbDocs", 10);
//    Node combineNode = new Node("combine", Node.cloneNodeList(original.getInternalNodes()));
//    ArrayList<ScoredDocument> initialResults = new ArrayList<ScoredDocument>();
//
//    // Only get as many as we need
//    Parameters localParameters = globalParameters.clone();
//    localParameters.set("requested", fbDocs);
//    combineNode = retrieval.transformQuery(combineNode, localParameters);
//    initialResults.addAll(Arrays.asList(retrieval.runQuery(combineNode, localParameters)));
//
//    // while that's running, extract the feedback parameters
//    int fbTerms = (int) parameters.get("fbTerms", 10);
//    Parameters tsvParameters = globalParameters.clone();
//    tsvParameters.set("fbDocs", fbDocs);
//    tsvParameters.set("parts", availableParts);
//    TermSelectionValueModel tsvModel = new TermSelectionValueModel(tsvParameters, retrieval, retrieval.getCollectionStatistics("#lengths:part=lengths()"));
//    tsvModel.initialize();
//
//    
//    Set<String> stopwords = WordLists.getWordList("rmstop");
//    Set<String> queryTerms = StructuredQuery.findQueryTerms(combineNode, Collections.singleton("extents"));
//
//    // Start constructing the final query
//    ArrayList<Node> newChildren = new ArrayList<Node>();
//    newChildren.addAll(original.getInternalNodes());
//
//    // Now we wait for the query to finish
//    Node newRoot = null;
//    Node expansionNode = tsvModel.generateExpansionQuery(initialResults, fbTerms, queryTerms, stopwords);
//    tsvModel.cleanup();
//
//    // The easiest thing to do really is extract the children and combine them w/ the existing
//    // query nodes, b/c the expansion is unweighted and flat.
//    newChildren.addAll(expansionNode.getInternalNodes());
//    newRoot = new Node("combine", new NodeParameters(), Node.cloneNodeList(newChildren), original.getPosition());
//    return newRoot;
  }
  
  public void beforeNode(Node object, Parameters qp) throws Exception {
    // do nothing
  }
}
