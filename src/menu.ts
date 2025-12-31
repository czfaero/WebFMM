import 'bootstrap';


export function InitMenu(param: any) {
    document.addEventListener('DOMContentLoaded', function () {


        const select_language = document.getElementById('select-language') as HTMLSelectElement;
        const button_start = document.getElementById('button-start') as HTMLButtonElement;
        const button_next_step = document.getElementById('button-next-step') as HTMLButtonElement;
        const button_pause = document.getElementById('button-pause') as HTMLButtonElement;
        const select_data = document.getElementById('select-data') as HTMLSelectElement;

        const log_slot0 = document.getElementById('log-slot0');
        const log_slot1 = document.getElementById('log-slot1');
        const log_slot2 = document.getElementById('log-slot2');
        const Log_slot0 = (msg) => log_slot0.innerHTML = msg;
        const Log_slot1 = (msg) => log_slot1.innerHTML = msg;
        const Log_slot2 = (msg) => log_slot2.innerHTML = msg;

        select_language.addEventListener('change', function (event) {
            const v = select_language.value;
            console.log("Change language: " + v);
            SetText(v);
        });
        select_language.value = AutoLanguage();
        select_language.dispatchEvent(new Event("change"));


        // before data
        if (!param.webgpuAvalible) {
            document.getElementById('msg-unavalible').setAttribute("active", "");
            const elements = document.querySelectorAll(".menu");
            elements.forEach(element => {
                element.setAttribute("disabled", "");
            });
            return;
        }


        for (const [key, value] of Object.entries(param.dataSelector)) {
            const op = document.createElement("option");
            op.value = key;
            op.innerHTML = (value as any).name;
            select_data.appendChild(op);
        }
        select_data.value = (select_data.children[0] as HTMLOptionElement).value;
        select_data.onchange = async function () {
            const count = await param.dataSelectorChange(select_data.value);
            Log_slot0("Node count: " + count);
            button_start.removeAttribute("disabled");
        }
        select_data.dispatchEvent(new Event("change"));


        // Data
        button_start.onclick = function () {
            param.StartUp(Log_slot0, Log_slot1, Log_slot2);
            button_start.setAttribute("disabled", "");
        }
        button_next_step.onclick = param.NextStep;
        button_pause.onclick = param.Pause;
        if (param.autoStart) {
            param.StartUp(Log_slot0, Log_slot1, Log_slot2);
        }

        // performance test
        const button_pt_all = document.querySelector("#pt-all");
        const button_pt_direct_cpu = document.querySelector("#pt-direct-cpu .emoji-button");
        const button_pt_direct_gpu = document.querySelector("#pt-direct-gpu .emoji-button");
        const button_pt_fmm_cpu = document.querySelector("#pt-fmm-cpu .emoji-button");
        const button_pt_fmm_gpu = document.querySelector("#pt-fmm-gpu .emoji-button");
        const slot_pt_direct_cpu = document.querySelector("#pt-direct-cpu .data-slot");
        const slot_pt_direct_gpu = document.querySelector("#pt-direct-gpu .data-slot");
        const slot_pt_fmm_cpu = document.querySelector("#pt-fmm-cpu .data-slot");
        const slot_pt_fmm_gpu = document.querySelector("#pt-fmm-gpu .data-slot");
        const slot_pt_task = document.querySelector("#pt-task .data-slot");


        const InitTaskUIEvent = (name, value, result_slot) => {
            const item_task = document.createElement("span");
            item_task.innerHTML = name;
            slot_pt_task.appendChild(item_task);
            // addPTTask( value, onTaskStart, onTaskEnd)
            param.addPTTask(value,
                () => { item_task.classList.add("running") },
                (task) => {
                    slot_pt_task.removeChild(item_task);
                    const item_result = document.createElement("span");
                    item_result.innerHTML = task.time.toFixed(1);
                    result_slot.appendChild(item_result);
                });
        }
        button_pt_direct_cpu.addEventListener("click", function () {
            InitTaskUIEvent("Direct CPU", "direct-cpu", slot_pt_direct_cpu);
        });
        button_pt_direct_gpu.addEventListener("click", function () {
            InitTaskUIEvent("Direct GPU", "direct-gpu", slot_pt_direct_gpu);
        });
        button_pt_fmm_cpu.addEventListener("click", function () {
            InitTaskUIEvent("FMM CPU", "fmm-cpu", slot_pt_fmm_cpu);
        });
        button_pt_fmm_gpu.addEventListener("click", function () {
            InitTaskUIEvent("FMM GPU", "fmm-gpu", slot_pt_fmm_gpu);
        });
        button_pt_all.addEventListener("click", function () {
            //InitTaskUIEvent("Direct CPU", "direct-cpu", slot_pt_direct_cpu);
            //InitTaskUIEvent("FMM CPU", "fmm-cpu", slot_pt_fmm_cpu);
            for (let i = 0; i < 10; i++) {
                InitTaskUIEvent("Direct GPU", "direct-gpu", slot_pt_direct_gpu);
            }
            for (let i = 0; i < 10; i++) {
                InitTaskUIEvent("FMM GPU", "fmm-gpu", slot_pt_fmm_gpu);
            }
        });


    });
}

function AutoLanguage() {
    const supported = ["ja", "zh"];
    let lang = navigator.language.substring(0, 2);
    if (!supported.includes(lang)) { lang = "en"; }
    return lang;
}

function SetText(lang: string) {
    const attr = `data-text-${lang}`;
    const elements = document.querySelectorAll(`[${attr}]`);
    elements.forEach(element => {
        element.innerHTML = element.getAttribute(attr).replaceAll("\\N", "<br>　　");
    });
}