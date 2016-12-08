// BSD License (http://lemurproject.org/galago-license)
package org.lemurproject.galago.core.parse.stem;

import org.tartarus.snowball.ext.russianStemmer;

/**
 *
 * @author fedorn
 * Use with: --stemmer+porter --stemmerClass/porter=org.lemurproject.galago.core.parse.stem.RussianStemmer
 */
public class RussianStemmer extends Stemmer {

  russianStemmer stemmer = new russianStemmer();

  @Override
  protected String stemTerm(String term) {
    String stem = term;
    stemmer.setCurrent(term);
    if (stemmer.stem()) {
      stem = stemmer.getCurrent();
    }
    return stem.toLowerCase();
  }
}
