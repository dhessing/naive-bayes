(ns naive-bayes.core-test
  (:require #?(:clj [clojure.test :refer :all]
               :cljs [cljs.test :refer-macros [deftest is testing run-tests]])
    [naive-bayes.core :as nb]))

(def spam-observations
  [{:class ["spam"] :word ["offer" "is" "secret"]}
   {:class ["spam"] :word ["click" "secret" "link"]}
   {:class ["spam"] :word ["secret" "sport" "link"]}
   {:class ["ham"] :word ["play" "sport" "today"]}
   {:class ["ham"] :word ["went" "play" "sport"]}
   {:class ["ham"] :word ["secret" "sport" "event"]}
   {:class ["ham"] :word ["sport" "is" "today"]}
   {:class ["ham"] :word ["sport" "costs" "money"]}])

(def spam-data
  [{:class {"spam" 3}, :word {"offer" 1, "is" 1, "secret" 3, "click" 1, "link" 2, "sport" 1}}
   {:class {"ham" 5},
    :word  {"went" 1, "play" 2, "today" 2, "is" 1, "event" 1, "sport" 5, "secret" 1, "money" 1, "costs" 1}}])

(deftest test-parse
  (is (= (nb/parse spam-observations) spam-data)))

(deftest test-p
  (is (= (nb/p spam-data 0 :class "spam") (/ 3 8)))
  (is (= (nb/p spam-data 0 :word "secret") (/ 1 6))))

(deftest test-p-given-class
  (is (= (nb/p-given-class spam-data 0 :word "secret" :class "spam") (/ 1 3)))
  (is (= (nb/p-given-class spam-data 0 :word "secret" :class "ham") (/ 1 15))))

(deftest test-p-given-feature
  (is (= (nb/p-given-feature spam-data :class "spam" :word "sport") (/ 1 6))))

(deftest test-bayes
  (is (= (nb/naive-bayes spam-data 0 :class "spam" :word "secret" :word "is" :word "secret") (/ 25 26)))
  (is (= (nb/naive-bayes spam-data 0 :class "spam" :word "today" :word "is" :word "secret") 0)))

(deftest test-classify
  (is (= (nb/classify spam-data 0 [:word "secret" :word "is" :word "secret"]) [:class "spam"]))
  (is (= (nb/classify spam-data 0 [:word "sport" :word "is" :word "today"]) [:class "ham"])))

(deftest test-classify-text
  (is (= (nb/classify-text spam-data 0 :word "secret is secret") [:class "spam"])))

(deftest test-laplace
  (is (= (nb/p spam-data 1 :class "spam") (/ 2 5)))
  (is (= (nb/p spam-data 1 :class "ham") (/ 3 5)))
  (is (= (nb/p-given-class spam-data 1 :word "today" :class "spam") (/ 1 21)))
  (is (= (nb/p-given-class spam-data 1 :word "today" :class "ham") (/ 1 9)))
  (is (= (nb/naive-bayes spam-data 1 :class "spam" :word "today" :word "is" :word "secret") (/ 324 667))))