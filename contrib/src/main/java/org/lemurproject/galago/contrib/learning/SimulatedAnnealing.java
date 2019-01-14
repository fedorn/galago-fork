/*
 *  BSD License (http://www.galagosearch.org/license)
 */
package org.lemurproject.galago.contrib.learning;

import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.utility.Parameters;

import java.util.*;

import static java.lang.Math.abs;
import static java.lang.Math.log;

/**
 * Nelder-Mead (or downhill simplex) method learning algorithm.
 * Based on code from section 10.5 of [1].
 * <p>
 * [1] Press, William H. Numerical recipes 3rd edition: The art of scientific computing. Cambridge university press, 2007.
 *
 * @author fsqcds
 */
public class SimulatedAnnealing extends Learner {
    private int NMAX = 50;
    private static final double TINY = 1.0e-10;

    protected Map<String, Double> dels;
    protected double del; // delta used for simplex initialization. See eq. 10.5.1 from [1].
    protected double ftol;
    protected double startT;
    protected double epsilonT;

    double yb = Double.NEGATIVE_INFINITY;
    Parameters pb;
    double ylo;
    int iter;
    List<Parameters> p;
    double[] y;
    Random ran = new Random();

    public SimulatedAnnealing(Parameters p, Retrieval r) throws Exception {
        super(p, r);

        this.NMAX = p.get("nmax", this.NMAX);

        this.dels = new HashMap<>();
        this.del = p.get("del", 0.05);
        this.ftol = p.get("ftol", 0.000001);
        this.startT = p.get("startt", 0.01);
        this.epsilonT = p.get("epsilont", 0.1);

        Parameters specialDels = Parameters.create();
        if (p.isMap("specialDel")) {
            specialDels = p.getMap("specialDel");
        }
        for (String param : learnableParameters.getParams()) {
            dels.put(param, specialDels.get(param, this.del));
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
                RetrievalModelInstance s = runSimulatedAnnealing(settingsInstance);
                double score = evaluate(s);
                s.setAnnotation("score", Double.toString(score));
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

    public RetrievalModelInstance runSimulatedAnnealing(RetrievalModelInstance parameterSettings) throws Exception {

        outputTraceStream.println(String.format("Initial parameter weights: %s Metric: %f. Starting optimization...", parameterSettings.toParameters().toString(), this.evaluate(parameterSettings)));

        List<String> params = new ArrayList<String>(this.learnableParameters.getParams());

        final int ndim = this.learnableParameters.getCount();
        final int mpts = ndim + 1;
        y = new double[mpts];

        p = new ArrayList<>();
        p.add(parameterSettings.toParameters());
        for (int i = 1; i < ndim + 1; i++) {
            Parameters inst = parameterSettings.toParameters();
            String paramName = params.get(i - 1);
            inst.set(paramName, inst.getDouble(paramName) + dels.get(paramName));
            p.add(inst);
            y[i] = evaluate(new RetrievalModelInstance(this.learnableParameters, p.get(i)));
            outputTraceStream.println(String.format("Initial point (%d) ... Metric: %f.", i, y[i]));
        }

        double temperature = startT;
        iter = NMAX;

        while (true) {
            boolean converged = anneal(temperature);

            if (converged) {
                outputTraceStream.println(String.format("Reached ftol... Done optimizing."));
                outputTraceStream.println(String.format("Best metric achieved: %s", yb));
                outputTraceStream.flush();
                return new RetrievalModelInstance(this.learnableParameters, pb);
            }

            // set temperature and iter according to annealing schedule
            temperature = (1 - epsilonT) * temperature;
            iter = NMAX;
        }
    }

    private boolean anneal(double temperature) throws Exception {
        final int ndim = this.learnableParameters.getCount();
        final int mpts = ndim + 1;

        Map<String, Double> psum = new HashMap<>();
        double tt = -temperature;
        getPsum(p, psum);
        while (true) {
            int ihi = 0;
            int ilo = 1;
            double yhi = y[0] - tt * log(ran.nextDouble());
            double ynlo = yhi;
            ylo = y[1] - tt * log(ran.nextDouble());
            if (yhi < ylo) {
                ilo = 0;
                ihi = 1;
                ynlo = ylo;
                ylo = yhi;
                yhi = ynlo;
            }

            for (int i = 3; i <= mpts; ++i) {
                double yt = y[i - 1] - tt * log(ran.nextDouble());
                if (yt >= yhi) {
                    ihi = i - 1;
                    yhi = yt;
                }
                if (yt < ylo) {
                    ynlo = ylo;
                    ilo = i - 1;
                    ylo = yt;
                } else if (yt < ynlo) {
                    ynlo = yt;
                }
            }
            double rtol = 2.0 * abs(ylo - yhi) / (abs(ylo) + abs(yhi));

            if (rtol < ftol || iter < 0) {
                double ty = y[0];
                y[0] = y[ihi];
                y[ihi] = ty;

                Parameters t = p.get(0);
                p.set(0, p.get(ihi));
                p.set(ihi, t);

                if (rtol < ftol) {
                    return true;
                } else {
                    return false;
                }
            }
            iter -= 2;
            double ytry = amotsa(p, y, psum, ilo, -1.0, tt);
            if (ytry >= ylo) {
                amotsa(p, y, psum, ilo, 2.0, tt);
            } else if (ytry <= ynlo) {
                double ysave = ylo;
                ytry = amotsa(p, y, psum, ilo, 0.5, tt);
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
                    iter -= ndim;
                    getPsum(p, psum);
                }
            } else ++iter;
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

    private double amotsa(List<Parameters> p, double[] y, Map<String, Double> psum, final int ilo, final double fac, final double tt) throws Exception {
        final int ndim = this.learnableParameters.getCount();
        Parameters ptry = p.get(0).clone();
        double fac1 = (1.0 - fac) / ndim;
        double fac2 = fac1 - fac;
        for (String param : this.learnableParameters.getParams()) {
            ptry.set(param, psum.get(param) * fac1 - p.get(ilo).getDouble(param) * fac2);
        }
        double ytry = evaluate(new RetrievalModelInstance(this.learnableParameters, ptry));
        outputTraceStream.println(String.format("ytry, ilo = %d, fac = %f, tt = %f ... Metric: %f.", ilo, fac, tt, ytry));
        if (ytry >= yb) {
            pb = ptry.clone();
            yb = ytry;
        }
        double yflu = ytry + tt * log(random.nextDouble());
        if (yflu > ylo) {
            y[ilo] = ytry;
            ylo = yflu;
            for (String param : this.learnableParameters.getParams()) {
                psum.put(param, psum.get(param) + ptry.getDouble(param) - p.get(ilo).getDouble(param));
            }
            p.set(ilo, ptry);
        }
        return yflu;
    }
}
