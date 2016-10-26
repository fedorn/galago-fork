/*
 *  BSD License (http://www.galagosearch.org/license)
 */
package org.lemurproject.galago.contrib.learning;

import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.utility.Parameters;

import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static java.lang.Math.abs;

/**
 * Nelder-Mead (or downhill simplex) method learning algorithm.
 * Based on code from section 10.5 of [1].
 * <p>
 * [1] Press, William H. Numerical recipes 3rd edition: The art of scientific computing. Cambridge university press, 2007.
 *
 * @author fsqcds
 */
public class NedlerMeadLearner extends Learner {
    private int NMAX = 5000; // maximum allowed number of metric evaluations
    private static final double TINY = 1.0e-10;

    protected Map<String, Double> minStepSizes;
    protected double minStepSize; // delta used for simplex initialization. See eq. 10.5.1 from [1].
    protected double ftol;

    public NedlerMeadLearner(Parameters p, Retrieval r) throws Exception {
        super(p, r);

        this.NMAX = p.get("nmax", this.NMAX);

        this.minStepSizes = new HashMap<>();
        this.minStepSize = p.get("minStepSize", 0.02);
        this.ftol = p.get("ftol", 0.000001);
        Parameters specialMinStepSizes = Parameters.create();
        if (p.isMap("specialMinStepSize")) {
            specialMinStepSizes = p.getMap("specialMinStepSize");
        }
        for (String param : learnableParameters.getParams()) {
            minStepSizes.put(param, specialMinStepSizes.get(param, this.minStepSize));
        }
    }

    @Override
    public RetrievalModelInstance learn() throws Exception {
        // loop for each random restart:
        final List<RetrievalModelInstance> learntParams = Collections.synchronizedList(new ArrayList<RetrievalModelInstance>());

        for (int i = 0; i < restarts; i++) {
            final RetrievalModelInstance settingsInstance;
            if (initialSettings.size() > i) {
                settingsInstance = initialSettings.get(i).clone();
            } else {
                settingsInstance = generateRandomInitalValues();
            }
            settingsInstance.setAnnotation("name", name + "-restart-" + i);
            try {
                double newScore, oldScore;
                RetrievalModelInstance s = settingsInstance.clone();
                newScore = evaluate(s);
                do {
                    s = runNedlerMead(s);
                    oldScore = newScore;
                    newScore = evaluate(s);
                } while (2.0 * abs(newScore - oldScore) / (abs(newScore) + abs(oldScore) + TINY) >= ftol);
                s.setAnnotation("score", Double.toString(newScore));
                learntParams.add(s);
                synchronized (outputPrintStream) {
                    outputPrintStream.println(s.toString());
                    outputPrintStream.flush();
                }
            } catch (Exception e) {
                System.err.println("Caught exception: \n" + e.toString());
                e.printStackTrace();
                synchronized (outputTraceStream) {
                    outputTraceStream.println(e.toString());
                    outputTraceStream.flush();
                }
            }
        }

        // check if we have learnt some values
        if (learntParams.isEmpty()) {
            return generateRandomInitalValues();
        } else {
            RetrievalModelInstance best = learntParams.get(0);
            double bestScore = Double.parseDouble(best.getAnnotation("score"));

            for (RetrievalModelInstance inst : learntParams) {
                double score = Double.parseDouble(inst.getAnnotation("score"));
                if (bestScore < score) {
                    best = inst;
                    bestScore = score;
                }
            }

            best.setAnnotation("name", name + "-best");

            outputPrintStream.println(best.toString());
            outputPrintStream.flush();

            return best;
        }
    }

    public RetrievalModelInstance runNedlerMead(RetrievalModelInstance parameterSettings) throws Exception {

        double best = this.evaluate(parameterSettings);
        outputTraceStream.println(String.format("Initial parameter weights: %s Metric: %f. Starting optimization...", parameterSettings.toParameters().toString(), best));

        List<String> params = new ArrayList<String>(this.learnableParameters.getParams());

        final int ndim = this.learnableParameters.getCount();
        final int mpts = ndim + 1;
        List<Parameters> p = new ArrayList<>();
        p.add(parameterSettings.toParameters());
        for (int i = 1; i < ndim + 1; i++) {
            Parameters inst = parameterSettings.toParameters();
            String paramName = params.get(i - 1);
            inst.set(paramName, inst.getDouble(paramName) + minStepSizes.get(paramName));
            p.add(inst);
        }

        Map<String, Double> psum = new HashMap<>();
        double[] y = new double[mpts];
        for (int i = 0; i < mpts; i++) {
            y[i] = evaluate(new RetrievalModelInstance(this.learnableParameters, p.get(i)));
            outputTraceStream.println(String.format("Initial point (%d) ... Metric: %f.", i, y[i]));
        }
        int nfunc = 0;
        int ihi, ilo, inlo;
        getPsum(p, psum);
        while (true) {
            ihi = 0;
            if (y[0] < y[1]) {
                inlo = 1;
                ilo = 0;
            } else {
                inlo = 0;
                ilo = 1;
            }
            for (int i = 0; i < mpts; i++) {
                if (y[i] >= y[ihi]) ihi = i;
                if (y[i] < y[ilo]) {
                    inlo = ilo;
                    ilo = i;
                } else if (y[i] < y[inlo] && i != ilo) inlo = i;
            }
            double rtol = 2.0 * abs(y[ihi] - y[ilo]) / (abs(y[ihi]) + abs(y[ilo]) + TINY);
            if (rtol < ftol) {
                outputTraceStream.println(String.format("Reached ftol... Done optimizing."));
                outputTraceStream.println(String.format("Best metric achieved: %s", best));
                outputTraceStream.flush();
                return new RetrievalModelInstance(this.learnableParameters, p.get(ihi));
            }

            if (nfunc >= NMAX) throw new RuntimeException("NMAX exceeded");
            nfunc += 2;
            double ytry = amotry(p, y, psum, ilo, -1.0);
            if (ytry >= y[ihi]) {
                amotry(p, y, psum, ilo, 2.0);
            } else if (ytry <= y[inlo]) {
                double ysave = y[ilo];
                ytry = amotry(p, y, psum, ilo, 0.5);
                if (ytry <= ysave) {
                    for (int i = 0; i < mpts; i++) {
                        if (i != ihi) {
                            for (String param : this.learnableParameters.getParams()) {
                                p.get(i).set(param, 0.5 * (p.get(i).getDouble(param) + p.get(ihi).getDouble(param)));
                            }
                            y[i] = evaluate(new RetrievalModelInstance(this.learnableParameters, p.get(i)));
                            outputTraceStream.println(String.format("After shrink point (%d) ... Metric: %f.", i, y[i]));
                        }
                    }
                    nfunc += ndim;
                    getPsum(p, psum);
                }
            } else --nfunc;
        }
    }

    private void getPsum(List<Parameters> p, Map<String, Double> psum) {
        for (String param : this.learnableParameters.getParams()) {
            double sum = 0.0;
            for (Parameters inst : p) {
                sum += inst.getDouble(param);
            }
            psum.put(param, sum);
        }
    }

    private double amotry(List<Parameters> p, double[] y, Map<String, Double> psum, final int ilo, final double fac) throws Exception {
        final int ndim = this.learnableParameters.getCount();
        Parameters ptry = p.get(0).clone();
        double fac1 = (1.0 - fac) / ndim;
        double fac2 = fac1 - fac;
        for (String param : this.learnableParameters.getParams()) {
            ptry.set(param, psum.get(param) * fac1 - p.get(ilo).getDouble(param) * fac2);
        }
        double ytry = evaluate(new RetrievalModelInstance(this.learnableParameters, ptry));
        outputTraceStream.println(String.format("ytry, ilo = %d, fac = %f ... Metric: %f.", ilo, fac, ytry));

        if (ytry > y[ilo]) {
            y[ilo] = ytry;
            for (String param : this.learnableParameters.getParams()) {
                psum.put(param, psum.get(param) + ptry.getDouble(param) - p.get(ilo).getDouble(param));
            }
            p.set(ilo, ptry);
        }
        return ytry;
    }
}
