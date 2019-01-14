package org.lemurproject.galago.contrib.retrieval.traversal;

import org.apache.commons.lang.math.NumberUtils;
import org.lemurproject.galago.core.index.stats.NodeStatistics;
import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.query.Node;
import org.lemurproject.galago.core.retrieval.query.NodeParameters;
import org.lemurproject.galago.core.retrieval.query.StructuredQuery;
import org.lemurproject.galago.core.util.TextPartAssigner;
import org.lemurproject.galago.utility.Parameters;

import java.util.ArrayList;
import java.util.List;

/**
 * FSDM with PRMS weighing of fields
 *
 * @author fedorn
 */
public class FSDMStarTraversal extends MLMTraversal {

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

    public FSDMStarTraversal(Retrieval retrieval) {
        super(retrieval);

        unigramDefault = globals.get("uniw", 0.8);
        orderedDefault = globals.get("odw", 0.1);
        unorderedDefault = globals.get("uww", 0.1);
        windowLimitDefault = (int) globals.get("windowLimit", 2);

        odOp = globals.get("sdm.od.op", "ordered");
        odWidth = (int) globals.get("sdm.od.width", 1);

        uwOp = globals.get("sdm.uw.op", "unordered");
        uwWidth = (int) globals.get("sdm.uw.width", 8);

    }

    @Override
    public Node afterNode(Node original, Parameters qp) throws Exception {
        if (original.getOperator().equals("fsdmstar")) {
            return buildSDMNode(original, qp);
        } else {
            return original;
        }
    }

    protected Node buildSDMNode(Node original, Parameters qp) throws Exception {
        NodeParameters np = original.getNodeParameters();
        List<Node> children = original.getInternalNodes();

        NodeParameters sdmWeights = new NodeParameters();
        List<Node> sdmNodes = new ArrayList<Node>();

        addUnigramNodes(original, qp, np, children, sdmWeights, sdmNodes);

        addBigramNodes(original, qp, np, children, sdmWeights, sdmNodes);

        return new Node("combine", sdmWeights, sdmNodes, original.getPosition());
    }

    @Override
    protected Node getUnigramNode(Node original, Parameters queryParameters, String term) throws Exception {
        String scorerType = queryParameters.get("scorer", globals.get("scorer", "dirichlet"));

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

//            double fieldWeight = 0.0;
//            if (fieldWeights != null && fieldWeights.containsKey(UNIGRAM_FIELD_PREFIX + field)) {
//                fieldWeight = fieldWeights.getDouble(UNIGRAM_FIELD_PREFIX + field);
//            } else {
//                fieldWeight = queryParameters.get(UNIGRAM_FIELD_PREFIX + field, 0.0);
//            }

            NodeStatistics ns = retrieval.getNodeStatistics(termFieldCounts);
            double fieldWeight = (double) ns.nodeFrequency / (double) fieldStats.getFieldStats().get(field).collectionLength; // P(t|F_j)

            nodeweights.set(Integer.toString(i), fieldWeight);
            normalizer += fieldWeight;

            Node termScore = new Node(scorerType);
            termScore.getNodeParameters().set("lengths", field);
            if (globals.containsKey("mu-" + field)) {
                termScore.getNodeParameters().set("mu", globals.getDouble("mu-" + field));
            }
            termScore.addChild(fieldStats.getFieldLenNodes().get(field).clone());
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

    protected void addBigramNodes(Node original, Parameters qp, NodeParameters np, List<Node> children, NodeParameters sdmWeights, List<Node> sdmNodes) throws Exception {
        int windowLimit = (int) qp.get("windowLimit", windowLimitDefault);
        windowLimit = (int) np.get("windowLimit", windowLimit);
        for (int n = 2; n <= windowLimit; n++) {
            for (int i = 0; i < (children.size() - n + 1); i++) {
                List<Node> termSequence = children.subList(i, i + n);
                double odWeight = computeWeight(termSequence, np, qp, true);
                BigramNodes bigramNodes = getBigramNodes(original, qp, termSequence);
                if (odWeight != 0) {
                    sdmWeights.set(Integer.toString(sdmNodes.size()), odWeight);
                    sdmNodes.add(bigramNodes.ordered);
                }
                double uwWeight = computeWeight(termSequence, np, qp, false);
                if (uwWeight != 0) {
                    sdmWeights.set(Integer.toString(sdmNodes.size()), uwWeight);
                    sdmNodes.add(bigramNodes.unordered);
                }
            }
        }
    }

    protected BigramNodes getBigramNodes(Node original, Parameters qp, List<Node> seq) throws Exception {
        NodeParameters np = original.getNodeParameters();

        NodeParameters orderedFieldWeights = new NodeParameters();
        double odNormalizer = 0.0;
        NodeParameters unwindowFieldWeights = new NodeParameters();
        double uwwNormalizer = 0.0;
//        for (int i = 0; i < fields.size(); i++) {
//            double odFieldWeight = 0.0;
//            double uwdFieldWeight = 0.0;
//            if (this.fieldWeights != null && this.fieldWeights.containsKey(ORDERED_FIELD_PREFIX + fields.get(i))) {
//                odFieldWeight = this.fieldWeights.getDouble(ORDERED_FIELD_PREFIX + fields.get(i));
//            } else {
//                odFieldWeight = qp.get(ORDERED_FIELD_PREFIX + fields.get(i), 0.0);
//            }
//            if (this.fieldWeights != null && this.fieldWeights.containsKey(UNWINDOW_FIELD_PREFIX + fields.get(i))) {
//                uwdFieldWeight = this.fieldWeights.getDouble(UNWINDOW_FIELD_PREFIX + fields.get(i));
//            } else {
//                uwdFieldWeight = qp.get(UNWINDOW_FIELD_PREFIX + fields.get(i), 0.0);
//            }
//            orderedFieldWeights.set(Integer.toString(i), odFieldWeight);
//            odNormalizer += odFieldWeight;
//            unwindowFieldWeights.set(Integer.toString(i), uwdFieldWeight);
//            uwwNormalizer += uwdFieldWeight;
//        }

        String scorerType = qp.get("scorer", globals.get("scorer", "dirichlet"));
        List<Node> orderedBigramFields = new ArrayList<Node>();
        List<Node> unorderedBigramFields = new ArrayList<Node>();
        for (int i = 0; i < fields.size(); i++) {
            String field = fields.get(i);
            Node orderedOperationNode = new Node(odOp, new NodeParameters(np.get("od.width", odWidth)));
            long unorderedWindow = np.get(("uw." + field + ".width"), np.get("uw.width", uwWidth));
            Node unorderedOperationNode = new Node(uwOp, new NodeParameters(unorderedWindow));
            for (Node t : seq) {
                String inFieldTerm = t.getNodeParameters().getAsSimpleString("default");
                if (NumberUtils.isNumber(inFieldTerm)) inFieldTerm = "@/" + inFieldTerm + "/";
                orderedOperationNode.addChild(TextPartAssigner.assignFieldPart(StructuredQuery.parse("#extents:" + inFieldTerm + "()"), this.retrieval.getAvailableParts(), field));
                unorderedOperationNode.addChild(TextPartAssigner.assignFieldPart(StructuredQuery.parse("#extents:" + inFieldTerm + "()"), this.retrieval.getAvailableParts(), field));
            }

            NodeStatistics odNs = retrieval.getNodeStatistics(orderedOperationNode);
            double odFieldWeight = (double) odNs.nodeFrequency / (double) fieldStats.getFieldStats().get(field).collectionLength; // P(t|F_j)
            orderedFieldWeights.set(Integer.toString(i), odFieldWeight);
            odNormalizer += odFieldWeight;

            NodeStatistics uwdNs = retrieval.getNodeStatistics(unorderedOperationNode);
            double uwdFieldWeight = (double) uwdNs.nodeFrequency / (double) fieldStats.getFieldStats().get(field).collectionLength; // P(t|F_j)
            unwindowFieldWeights.set(Integer.toString(i), uwdFieldWeight);
            uwwNormalizer += uwdFieldWeight;

            Node orderedBigramScore = new Node(scorerType);
            orderedBigramScore.getNodeParameters().set("lengths", field);
            if (globals.containsKey("mu-" + field)) {
                orderedBigramScore.getNodeParameters().set("mu", globals.getDouble("mu-" + field));
            }
            orderedBigramScore.addChild(fieldStats.getFieldLenNodes().get(field).clone());
            orderedBigramScore.addChild(orderedOperationNode);
            orderedBigramFields.add(orderedBigramScore);

            Node unorderedBigramScore = new Node(scorerType);
            unorderedBigramScore.getNodeParameters().set("lengths", field);
            if (globals.containsKey("mu-" + field)) {
                unorderedBigramScore.getNodeParameters().set("mu", globals.getDouble("mu-" + field));
            }
            unorderedBigramScore.addChild(fieldStats.getFieldLenNodes().get(field).clone());
            unorderedBigramScore.addChild(unorderedOperationNode);
            unorderedBigramFields.add(unorderedBigramScore);
        }

        // normalize field weights
        if (odNormalizer != 0) {
            for (int i = 0; i < fields.size(); i++) {
                String key = Integer.toString(i);
                orderedFieldWeights.set(key, orderedFieldWeights.getDouble(key) / odNormalizer);
            }
        } else {
            for (int i = 0; i < fields.size(); i++) {
                String key = Integer.toString(i);
                orderedFieldWeights.set(key, 1.0);
            }
        }
        if (uwwNormalizer != 0) {
            for (int i = 0; i < fields.size(); i++) {
                String key = Integer.toString(i);
                unwindowFieldWeights.set(key, unwindowFieldWeights.getDouble(key) / uwwNormalizer);
            }
        } else {
            for (int i = 0; i < fields.size(); i++) {
                String key = Integer.toString(i);
                unwindowFieldWeights.set(key, 1.0);
            }
        }

        Node orderedNode = new Node("wsum", orderedFieldWeights, orderedBigramFields);
        Node unorderedNode = new Node("wsum", unwindowFieldWeights, unorderedBigramFields);
        return new BigramNodes(orderedNode, unorderedNode);
    }

    protected double computeWeight(String term, NodeParameters np, Parameters qp) throws Exception {
        double unigramW = qp.get("uniw", unigramDefault);
        unigramW = np.get("uniw", unigramW);
        return unigramW;
    }

    protected double computeWeight(List<Node> bigram, NodeParameters np, Parameters qp, boolean isOrdered) throws Exception {
        if (isOrdered) {
            double orderedW = qp.get("odw", orderedDefault);
            orderedW = np.get("odw", orderedW);
            return orderedW;
        } else {
            double unorderedW = qp.get("uww", unorderedDefault);
            unorderedW = np.get("uww", unorderedW);
            return unorderedW;
        }
    }

    protected class BigramNodes {
        protected final Node ordered;
        protected final Node unordered;

        public BigramNodes(Node ordered, Node unordered) {
            this.ordered = ordered;
            this.unordered = unordered;
        }
    }

}
