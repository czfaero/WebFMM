import 'bootstrap';


export function InitMenu(param: any) {
    document.addEventListener('DOMContentLoaded', function () {


        const select_language = document.getElementById('select-language') as HTMLSelectElement;
        const button_start = document.getElementById('button-start') as HTMLButtonElement;
        const button_next_step = document.getElementById('button-next-step') as HTMLButtonElement;

        const msg_data = document.getElementById('msg-data');
        const msg_render = document.getElementById('msg-render');
        const Log_data = (msg) => msg_data.innerHTML = msg;
        const Log_render = (msg) => msg_render.innerHTML = msg;

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


        // Data
        button_start.onclick = function () {
            param.StartUp(Log_render, Log_data);
            button_start.setAttribute("disabled", "");
        }
        button_next_step.onclick = param.NextStep;
        if (param.autoStart) {
            param.StartUp(Log_render, Log_data);
        }
    });
}

function AutoLanguage() {
    let lang = "en";
    if (navigator.language.startsWith("zh")) { lang = "zh"; }
    switch (navigator.language) {
        case "ja-JP": lang = "ja"; break;
    }
    return lang;
}

function SetText(lang: string) {
    const attr = `data-text-${lang}`;
    const elements = document.querySelectorAll(`[${attr}]`);
    elements.forEach(element => {
        element.innerHTML = element.getAttribute(attr);
    });
}