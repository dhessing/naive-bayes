(ns naive-bayes.core-test
  (:require [clojure.test :refer :all]
            [naive-bayes.core :refer :all]))

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
  (is (= (parse spam-observations) spam-data)))

(deftest test-p
  (is (= (p spam-data :class "spam") (/ 3 8)))
  (is (= (p spam-data :word "secret") (/ 1 6))))

(deftest test-p-given-class
  (is (= (p-given-class spam-data :word "secret" :class "spam") (/ 1 3)))
  (is (= (p-given-class spam-data :word "secret" :class "ham") (/ 1 15))))

(deftest test-p-given-feature
  (is (= (p-given-feature spam-data :class "spam" :word "sport") (/ 1 6))))

(deftest test-bayes
  (is (= (naive-bayes spam-data :class "spam" :word "secret" :word "is" :word "secret") 25/26))
  (is (= (naive-bayes spam-data :class "spam" :word "today" :word "is" :word "secret") 0)))

(deftest test-classify
  (is (= (classify spam-data :word "secret" :word "is" :word "secret") [:class "spam"]))
  (is (= (classify spam-data :word "sport" :word "is" :word "today") [:class "ham"])))