let headache = {
  두통: [
    ["망치로 맞은 듯한 극심한 통증인가요?", 1, 2],
    ["분출성 구토가 있으신가요?", 3, 4],
    ["편두통이 있으신가요?", 5, 6],
    ["병원을 반드시 방문하세요"],
    [["내관", "합곡"]],
    ["구역감이 있으신가요?", 7, 8],
    ["분출성 구토가 있으신가요?", 9, 10],
    [["내관", "합곡"]],
    [["상양", "관충"]],
    ["병원을 반드시 방문하세요"],
    [["내관", "합곡"]]
  ],
  흉통: [
    ["음식 섭취와 관련이 있을까요?", 1, 2],
    [["합곡", "어제", "내관"]],
    ["스트레스 성이나 운동 후 발생인가요?", 3, 4],
    ["30분 미만 지속/가슴조이는 통증이라면 협심증입니다."],
    ["병원을 반드시 방문하세요 "],
    [["합곡", "어제", "내관"]]
  ]
};

let symptom = document.getElementById("symptom").getAttribute("value");
let question = document.getElementById("question");
// let prior_answer = document.getElementById("prior-answer");
let outer_scope_box = document.getElementById("outer_scope_box");
let inner_scope_box = document.getElementById("inner_scope_box");
let yes_q = document.getElementById("yes_q");
let no_q = document.getElementById("no_q");
let yes = document.getElementById("yes");
let no = document.getElementById("no");
let index = 0;
let size = headache[symptom][index].length;
let symp = headache[symptom][index];

yes_q.onclick = function () {
    outer_scope_box.style.display = "none";
    inner_scope_box.style.display = "block";
    // prior_answer.style.display = "none";
    question.innerHTML = symp[0];
};

no_q.onclick = function () {
    outer_scope_box.style.display = "none";
    inner_scope_box.style.display = "none";
};


yes.onclick = function () {
    index = symp[1];
    size = headache[symptom][index].length;
    symp = headache[symptom][index];

    if (size === 3) {
        question.innerHTML = symp[0];
    } else if (size === 1) {
        question.innerHTML = symp[0];
        inner_scope_box.style.display = "none";
    }
};

no.onclick = function () {
    index = symp[2];
    size = headache[symptom][index].length;
    symp = headache[symptom][index];

    if (size === 3) {
        question.innerHTML = symp[0];
    } else if (size === 1) {
        question.innerHTML = symp[0];
        inner_scope_box.style.display = "none";
    }
};