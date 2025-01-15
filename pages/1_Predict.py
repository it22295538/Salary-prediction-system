import streamlit as st
import pickle
import numpy as np
import sklearn

# load the saved model
with open('./best__model.pkl', 'rb') as file:
    best_model = pickle.load(file)


def yearsCodeProEncoder(x):
    return (x - 1) / 27

def workExpEncoder(x):
    return x / 30

mapping_json = {
    "Country": {
    "0": "australia",
    "1": "brazil",
    "2": "canada",
    "3": "france",
    "4": "germany",
    "5": "india",
    "6": "italy",
    "7": "netherlands",
    "8": "poland",
    "9": "united kingdom of great britain and northern ireland",
    "10": "united states of america"
    },

    "OpSysProfessional use": {
        "0": "Other",
        "1": "android;ios;ipados;macos;windows",
        "2": "android;ios;macos",
        "3": "android;ios;macos;windows",
        "4": "android;macos",
        "5": "android;ubuntu",
        "6": "android;ubuntu;windows",
        "7": "android;ubuntu;windows;windows subsystem for linux (wsl)",
        "8": "android;windows",
        "9": "android;windows;windows subsystem for linux (wsl)",
        "10": "arch",
        "11": "arch;macos",
        "12": "arch;ubuntu",
        "13": "debian",
        "14": "debian;macos",
        "15": "debian;macos;ubuntu",
        "16": "debian;ubuntu",
        "17": "debian;ubuntu;windows",
        "18": "debian;ubuntu;windows;windows subsystem for linux (wsl)",
        "19": "debian;windows",
        "20": "debian;windows;windows subsystem for linux (wsl)",
        "21": "fedora",
        "22": "ios",
        "23": "ios;ipados;macos",
        "24": "ios;macos",
        "25": "ios;macos;ubuntu",
        "26": "macos",
        "27": "macos;other linux-based",
        "28": "macos;other linux-based;ubuntu",
        "29": "macos;red hat",
        "30": "macos;ubuntu",
        "31": "macos;ubuntu;windows",
        "32": "macos;ubuntu;windows subsystem for linux (wsl)",
        "33": "macos;ubuntu;windows;windows subsystem for linux (wsl)",
        "34": "macos;windows",
        "35": "macos;windows subsystem for linux (wsl)",
        "36": "macos;windows;windows subsystem for linux (wsl)",
        "37": "other (please specify):",
        "38": "other linux-based",
        "39": "other linux-based;other (please specify):",
        "40": "other linux-based;ubuntu",
        "41": "other linux-based;ubuntu;windows;windows subsystem for linux (wsl)",
        "42": "other linux-based;windows",
        "43": "other linux-based;windows;windows subsystem for linux (wsl)",
        "44": "red hat",
        "45": "red hat;windows",
        "46": "red hat;windows;windows subsystem for linux (wsl)",
        "47": "ubuntu",
        "48": "ubuntu;windows",
        "49": "ubuntu;windows subsystem for linux (wsl)",
        "50": "ubuntu;windows;windows subsystem for linux (wsl)",
        "51": "windows",
        "52": "windows subsystem for linux (wsl)",
        "53": "windows;windows subsystem for linux (wsl)"
    },

    "ProfessionalTech": {
        "0": "Other",
        "1": "ai-assisted technology tool(s)",
        "2": "automated testing",
        "3": "automated testing;continuous integration (ci) and (more often) continuous delivery",
        "4": "automated testing;continuous integration (ci) and (more often) continuous delivery;ai-assisted technology tool(s)",
        "5": "automated testing;developer portal or other central places to find tools/services",
        "6": "automated testing;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery",
        "7": "automated testing;observability tools",
        "8": "automated testing;observability tools;continuous integration (ci) and (more often) continuous delivery",
        "9": "automated testing;observability tools;continuous integration (ci) and (more often) continuous delivery;ai-assisted technology tool(s)",
        "10": "automated testing;observability tools;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery",
        "11": "continuous integration (ci) and (more often) continuous delivery",
        "12": "continuous integration (ci) and (more often) continuous delivery;ai-assisted technology tool(s)",
        "13": "developer portal or other central places to find tools/services",
        "14": "developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery",
        "15": "devops function",
        "16": "devops function;automated testing",
        "17": "devops function;automated testing;continuous integration (ci) and (more often) continuous delivery",
        "18": "devops function;automated testing;continuous integration (ci) and (more often) continuous delivery;ai-assisted technology tool(s)",
        "19": "devops function;automated testing;developer portal or other central places to find tools/services",
        "20": "devops function;automated testing;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery",
        "21": "devops function;automated testing;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery;ai-assisted technology tool(s)",
        "22": "devops function;automated testing;innersource initiative;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery",
        "23": "devops function;automated testing;observability tools",
        "24": "devops function;automated testing;observability tools;continuous integration (ci) and (more often) continuous delivery",
        "25": "devops function;automated testing;observability tools;continuous integration (ci) and (more often) continuous delivery;ai-assisted technology tool(s)",
        "26": "devops function;automated testing;observability tools;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery",
        "27": "devops function;automated testing;observability tools;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery;ai-assisted technology tool(s)",
        "28": "devops function;automated testing;observability tools;innersource initiative;continuous integration (ci) and (more often) continuous delivery",
        "29": "devops function;automated testing;observability tools;innersource initiative;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery",
        "30": "devops function;continuous integration (ci) and (more often) continuous delivery",
        "31": "devops function;continuous integration (ci) and (more often) continuous delivery;ai-assisted technology tool(s)",
        "32": "devops function;developer portal or other central places to find tools/services",
        "33": "devops function;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery",
        "34": "devops function;microservices",
        "35": "devops function;microservices;automated testing",
        "36": "devops function;microservices;automated testing;continuous integration (ci) and (more often) continuous delivery",
        "37": "devops function;microservices;automated testing;continuous integration (ci) and (more often) continuous delivery;ai-assisted technology tool(s)",
        "38": "devops function;microservices;automated testing;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery",
        "39": "devops function;microservices;automated testing;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery;ai-assisted technology tool(s)",
        "40": "devops function;microservices;automated testing;innersource initiative;continuous integration (ci) and (more often) continuous delivery",
        "41": "devops function;microservices;automated testing;innersource initiative;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery",
        "42": "devops function;microservices;automated testing;innersource initiative;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery;ai-assisted technology tool(s)",
        "43": "devops function;microservices;automated testing;observability tools",
        "44": "devops function;microservices;automated testing;observability tools;continuous integration (ci) and (more often) continuous delivery",
        "45": "devops function;microservices;automated testing;observability tools;continuous integration (ci) and (more often) continuous delivery;ai-assisted technology tool(s)",
        "46": "devops function;microservices;automated testing;observability tools;developer portal or other central places to find tools/services",
        "47": "devops function;microservices;automated testing;observability tools;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery",
        "48": "devops function;microservices;automated testing;observability tools;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery;ai-assisted technology tool(s)",
        "49": "devops function;microservices;automated testing;observability tools;innersource initiative;continuous integration (ci) and (more often) continuous delivery",
        "50": "devops function;microservices;automated testing;observability tools;innersource initiative;continuous integration (ci) and (more often) continuous delivery;ai-assisted technology tool(s)",
        "51": "devops function;microservices;automated testing;observability tools;innersource initiative;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery",
        "52": "devops function;microservices;automated testing;observability tools;innersource initiative;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery;ai-assisted technology tool(s)",
        "53": "devops function;microservices;continuous integration (ci) and (more often) continuous delivery",
        "54": "devops function;microservices;continuous integration (ci) and (more often) continuous delivery;ai-assisted technology tool(s)",
        "55": "devops function;microservices;developer portal or other central places to find tools/services",
        "56": "devops function;microservices;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery",
        "57": "devops function;microservices;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery;ai-assisted technology tool(s)",
        "58": "devops function;microservices;observability tools",
        "59": "devops function;microservices;observability tools;continuous integration (ci) and (more often) continuous delivery",
        "60": "devops function;microservices;observability tools;continuous integration (ci) and (more often) continuous delivery;ai-assisted technology tool(s)",
        "61": "devops function;microservices;observability tools;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery",
        "62": "devops function;observability tools;continuous integration (ci) and (more often) continuous delivery",
        "63": "devops function;observability tools;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery",
        "64": "microservices",
        "65": "microservices;automated testing",
        "66": "microservices;automated testing;continuous integration (ci) and (more often) continuous delivery",
        "67": "microservices;automated testing;continuous integration (ci) and (more often) continuous delivery;ai-assisted technology tool(s)",
        "68": "microservices;automated testing;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery",
        "69": "microservices;automated testing;observability tools;continuous integration (ci) and (more often) continuous delivery",
        "70": "microservices;automated testing;observability tools;continuous integration (ci) and (more often) continuous delivery;ai-assisted technology tool(s)",
        "71": "microservices;automated testing;observability tools;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery",
        "72": "microservices;continuous integration (ci) and (more often) continuous delivery",
        "73": "microservices;continuous integration (ci) and (more often) continuous delivery;ai-assisted technology tool(s)",
        "74": "microservices;developer portal or other central places to find tools/services",
        "75": "microservices;developer portal or other central places to find tools/services;continuous integration (ci) and (more often) continuous delivery",
        "76": "microservices;observability tools;continuous integration (ci) and (more often) continuous delivery",
        "77": "none of these",
        "78": "observability tools",
        "79": "observability tools;continuous integration (ci) and (more often) continuous delivery"
    },

    "DevType": {
        "0": "Other",
        "1": "academic researcher",
        "2": "blockchain",
        "3": "cloud infrastructure engineer",
        "4": "data or business analyst",
        "5": "data scientist or machine learning specialist",
        "6": "developer advocate",
        "7": "developer experience",
        "8": "developer, back-end",
        "9": "developer, desktop or enterprise applications",
        "10": "developer, embedded applications or devices",
        "11": "developer, front-end",
        "12": "developer, full-stack",
        "13": "developer, game or graphics",
        "14": "developer, mobile",
        "15": "developer, qa or test",
        "16": "devops specialist",
        "17": "engineer, data",
        "18": "engineer, site reliability",
        "19": "engineering manager",
        "20": "other (please specify):",
        "21": "product manager",
        "22": "project manager",
        "23": "research & development role",
        "24": "security professional",
        "25": "senior executive (c-suite, vp, etc.)",
        "26": "system administrator"
    },

    "Industry": {
        "0": "advertising services",
        "1": "financial services",
        "2": "healthcare",
        "3": "higher education",
        "4": "information services, it, software development, or other technology",
        "5": "insurance",
        "6": "legal services",
        "7": "manufacturing, transportation, or supply chain",
        "8": "oil & gas",
        "9": "other",
        "10": "retail and consumer services",
        "11": "wholesale"
    },

    "EdLevel": {
        "0": "Associate degree",
        "1": "Bachelor\u2019s degree",
        "2": "Less than a Bachelors",
        "3": "Master\u2019s degree",
        "4": "Primary/elementary school",
        "5": "Professional degree",
        "6": "Secondary school",
        "7": "Some college/university study without earning a degree"
    },

    "Age": {
        "0": "18-24 years old",
        "1": "25-34 years old",
        "2": "35-44 years old",
        "3": "45-54 years old",
        "4": "55-64 years old",
        "5": "Other"
    },

    "LanguageHaveWorkedWith": {
        "0": "Other",
        "1": "bash/shell (all shells);c#;html/css;javascript;powershell;python;sql;typescript",
        "2": "bash/shell (all shells);c#;html/css;javascript;powershell;sql;typescript",
        "3": "bash/shell (all shells);html/css;java;javascript;python;sql;typescript",
        "4": "bash/shell (all shells);html/css;java;javascript;sql;typescript",
        "5": "bash/shell (all shells);html/css;javascript;php;python;sql;typescript",
        "6": "bash/shell (all shells);html/css;javascript;php;sql",
        "7": "bash/shell (all shells);html/css;javascript;php;sql;typescript",
        "8": "bash/shell (all shells);html/css;javascript;python;sql",
        "9": "bash/shell (all shells);html/css;javascript;python;sql;typescript",
        "10": "bash/shell (all shells);html/css;javascript;python;typescript",
        "11": "bash/shell (all shells);html/css;javascript;sql;typescript",
        "12": "bash/shell (all shells);html/css;javascript;typescript",
        "13": "bash/shell (all shells);python",
        "14": "bash/shell (all shells);python;sql",
        "15": "c#",
        "16": "c#;html/css;javascript;powershell;python;sql;typescript",
        "17": "c#;html/css;javascript;powershell;sql",
        "18": "c#;html/css;javascript;powershell;sql;typescript",
        "19": "c#;html/css;javascript;powershell;typescript",
        "20": "c#;html/css;javascript;python;sql;typescript",
        "21": "c#;html/css;javascript;sql",
        "22": "c#;html/css;javascript;sql;typescript",
        "23": "c#;html/css;javascript;typescript",
        "24": "c#;javascript;powershell;sql;typescript",
        "25": "c#;javascript;sql",
        "26": "c#;javascript;sql;typescript",
        "27": "c#;javascript;typescript",
        "28": "c#;sql",
        "29": "html/css;java;javascript;python;sql;typescript",
        "30": "html/css;java;javascript;sql",
        "31": "html/css;java;javascript;sql;typescript",
        "32": "html/css;java;javascript;typescript",
        "33": "html/css;javascript",
        "34": "html/css;javascript;php",
        "35": "html/css;javascript;php;python;sql;typescript",
        "36": "html/css;javascript;php;sql",
        "37": "html/css;javascript;php;sql;typescript",
        "38": "html/css;javascript;php;typescript",
        "39": "html/css;javascript;python",
        "40": "html/css;javascript;python;sql",
        "41": "html/css;javascript;python;sql;typescript",
        "42": "html/css;javascript;python;typescript",
        "43": "html/css;javascript;ruby;sql",
        "44": "html/css;javascript;ruby;sql;typescript",
        "45": "html/css;javascript;sql;typescript",
        "46": "html/css;javascript;typescript",
        "47": "java",
        "48": "java;javascript;typescript",
        "49": "java;sql",
        "50": "javascript",
        "51": "javascript;python",
        "52": "javascript;python;sql;typescript",
        "53": "javascript;python;typescript",
        "54": "javascript;sql;typescript",
        "55": "javascript;typescript",
        "56": "python",
        "57": "python;sql",
        "58": "typescript"
    },

    "RemoteWork": {
        "0": "hybrid (some remote, some in-person)",
        "1": "in-person",
        "2": "remote"
    },

    "Employment": {
        "0": "Other",
        "1": "employed, full-time",
        "2": "employed, full-time;employed, part-time",
        "3": "employed, full-time;independent contractor, freelancer, or self-employed",
        "4": "employed, part-time",
        "5": "independent contractor, freelancer, or self-employed",
        "6": "independent contractor, freelancer, or self-employed;employed, part-time"
    },

    "ToolsTechHaveWorkedWith": {
        "0": "Other",
        "1": "composer",
        "2": "docker",
        "3": "docker;gradle;maven (build tool)",
        "4": "docker;homebrew",
        "5": "docker;homebrew;kubernetes",
        "6": "docker;homebrew;npm",
        "7": "docker;homebrew;npm;webpack;yarn",
        "8": "docker;homebrew;npm;yarn",
        "9": "docker;kubernetes",
        "10": "docker;kubernetes;maven (build tool)",
        "11": "docker;kubernetes;npm",
        "12": "docker;maven (build tool)",
        "13": "docker;maven (build tool);npm",
        "14": "docker;msbuild;npm;nuget;visual studio solution",
        "15": "docker;msbuild;nuget;visual studio solution",
        "16": "docker;npm",
        "17": "docker;npm;nuget",
        "18": "docker;npm;pip",
        "19": "docker;npm;webpack",
        "20": "docker;npm;webpack;yarn",
        "21": "docker;npm;yarn",
        "22": "docker;nuget",
        "23": "docker;pip",
        "24": "gradle",
        "25": "homebrew;npm",
        "26": "maven (build tool)",
        "27": "maven (build tool);npm",
        "28": "msbuild",
        "29": "msbuild;npm;nuget",
        "30": "msbuild;npm;nuget;visual studio solution",
        "31": "msbuild;nuget",
        "32": "msbuild;nuget;visual studio solution",
        "33": "msbuild;visual studio solution",
        "34": "npm",
        "35": "npm;nuget",
        "36": "npm;nuget;visual studio solution",
        "37": "npm;vite;webpack",
        "38": "npm;webpack",
        "39": "npm;webpack;yarn",
        "40": "npm;yarn",
        "41": "nuget",
        "42": "nuget;visual studio solution",
        "43": "pip",
        "44": "visual studio solution",
        "45": "webpack"
    },

    "DatabaseHaveWorkedWith": {
        "0": "Other",
        "1": "cloud firestore",
        "2": "cloud firestore;postgresql",
        "3": "cosmos db",
        "4": "cosmos db;microsoft sql server",
        "5": "cosmos db;microsoft sql server;redis",
        "6": "dynamodb",
        "7": "dynamodb;elasticsearch;postgresql",
        "8": "dynamodb;microsoft sql server",
        "9": "dynamodb;mysql",
        "10": "dynamodb;postgresql",
        "11": "dynamodb;postgresql;redis",
        "12": "elasticsearch",
        "13": "elasticsearch;microsoft sql server",
        "14": "elasticsearch;mongodb;postgresql",
        "15": "elasticsearch;mysql;postgresql",
        "16": "elasticsearch;mysql;postgresql;redis",
        "17": "elasticsearch;mysql;redis",
        "18": "elasticsearch;postgresql",
        "19": "elasticsearch;postgresql;redis",
        "20": "firebase realtime database",
        "21": "h2;postgresql",
        "22": "mariadb",
        "23": "mariadb;microsoft sql server;mysql",
        "24": "mariadb;mysql",
        "25": "mariadb;mysql;postgresql",
        "26": "mariadb;mysql;postgresql;redis",
        "27": "mariadb;mysql;postgresql;redis;sqlite",
        "28": "mariadb;mysql;postgresql;sqlite",
        "29": "mariadb;mysql;redis",
        "30": "mariadb;mysql;sqlite",
        "31": "mariadb;postgresql",
        "32": "microsoft sql server",
        "33": "microsoft sql server;mongodb",
        "34": "microsoft sql server;mongodb;mysql",
        "35": "microsoft sql server;mysql",
        "36": "microsoft sql server;mysql;postgresql",
        "37": "microsoft sql server;mysql;postgresql;sqlite",
        "38": "microsoft sql server;mysql;sqlite",
        "39": "microsoft sql server;oracle",
        "40": "microsoft sql server;postgresql",
        "41": "microsoft sql server;postgresql;redis",
        "42": "microsoft sql server;postgresql;sqlite",
        "43": "microsoft sql server;redis",
        "44": "microsoft sql server;sqlite",
        "45": "mongodb",
        "46": "mongodb;mysql",
        "47": "mongodb;mysql;postgresql",
        "48": "mongodb;mysql;postgresql;redis",
        "49": "mongodb;mysql;postgresql;sqlite",
        "50": "mongodb;postgresql",
        "51": "mongodb;postgresql;redis",
        "52": "mongodb;postgresql;sqlite",
        "53": "mongodb;redis",
        "54": "mongodb;sqlite",
        "55": "mysql",
        "56": "mysql;oracle",
        "57": "mysql;postgresql",
        "58": "mysql;postgresql;redis",
        "59": "mysql;postgresql;redis;sqlite",
        "60": "mysql;postgresql;sqlite",
        "61": "mysql;redis",
        "62": "mysql;sqlite",
        "63": "oracle",
        "64": "oracle;postgresql",
        "65": "postgresql",
        "66": "postgresql;redis",
        "67": "postgresql;redis;sqlite",
        "68": "postgresql;snowflake",
        "69": "postgresql;sqlite",
        "70": "postgresql;supabase",
        "71": "sqlite"
    },

    "WebframeHaveWorkedWith": {
        "0": "Other",
        "1": "angular",
        "2": "angular;asp.net core",
        "3": "angular;asp.net core;node.js",
        "4": "angular;asp.net;asp.net core",
        "5": "angular;asp.net;asp.net core;node.js",
        "6": "angular;express;node.js",
        "7": "angular;node.js",
        "8": "angular;node.js;spring boot",
        "9": "angular;react",
        "10": "angular;spring boot",
        "11": "asp.net",
        "12": "asp.net core",
        "13": "asp.net core;blazor",
        "14": "asp.net core;jquery",
        "15": "asp.net core;node.js;react",
        "16": "asp.net core;react",
        "17": "asp.net core;vue.js",
        "18": "asp.net;asp.net core",
        "19": "asp.net;asp.net core;blazor",
        "20": "asp.net;asp.net core;blazor;jquery",
        "21": "asp.net;asp.net core;jquery",
        "22": "asp.net;asp.net core;jquery;node.js;react",
        "23": "asp.net;asp.net core;jquery;react",
        "24": "asp.net;asp.net core;node.js;react",
        "25": "asp.net;asp.net core;react",
        "26": "asp.net;jquery",
        "27": "blazor",
        "28": "django",
        "29": "django;fastapi",
        "30": "django;fastapi;flask",
        "31": "django;flask",
        "32": "django;react",
        "33": "express;nestjs;next.js;node.js;react",
        "34": "express;next.js;node.js;react",
        "35": "express;node.js",
        "36": "express;node.js;react",
        "37": "fastapi",
        "38": "fastapi;flask",
        "39": "fastapi;react",
        "40": "flask",
        "41": "flask;node.js;react",
        "42": "flask;react",
        "43": "jquery",
        "44": "jquery;node.js",
        "45": "jquery;spring boot",
        "46": "jquery;wordpress",
        "47": "laravel",
        "48": "laravel;vue.js",
        "49": "next.js;node.js;react",
        "50": "next.js;react",
        "51": "node.js",
        "52": "node.js;react",
        "53": "node.js;react;spring boot",
        "54": "node.js;spring boot",
        "55": "phoenix",
        "56": "react",
        "57": "react;ruby on rails",
        "58": "react;spring boot",
        "59": "ruby on rails",
        "60": "spring boot",
        "61": "spring boot;vue.js",
        "62": "svelte",
        "63": "symfony",
        "64": "vue.js",
        "65": "wordpress"
    }
}

st.markdown('<h1 style="color: blue; text-align: center;">Let us Predict</h1>', unsafe_allow_html=True)
st.markdown('<p style="color: red; text-align: center;">Enter the following details for prediction &#128071;</p>', unsafe_allow_html=True)
st.markdown('<p style="color: red; text-align: center;">For options, Please select the best answer suits you!</p>', unsafe_allow_html=True)

# model input
features = []

# loop mapping JSON
for feature_name, options in mapping_json.items():
    # select options for each topic values
    def featureNameChange(feature_name):
        if feature_name == "Country":
            return "Country"
        if feature_name == "OpSysProfessional use":
            return "Operating Systems used"
        if feature_name == "ProfessionalTech":
            return "Professional Technologies used"
        if feature_name == "DevType":
            return "Developer Type"
        if feature_name == "Industry":
            return "Industry"
        if feature_name == "EdLevel":
            return "Education Level"
        if feature_name == "Age":
            return "Your Age"
        if feature_name == "LanguageHaveWorkedWith":
            return "Programming Languages Worked With"
        if feature_name == "RemoteWork":
            return "Work Type"
        if feature_name == "Employment":
            return "Employment Status"
        if feature_name == "ToolsTechHaveWorkedWith":
            return "Tools and Technologies Worked With"
        if feature_name == "DatabaseHaveWorkedWith":
            return "Databases Worked With"
        if feature_name == "WebframeHaveWorkedWith":
            return "Web Frameworks Worked With"

    all_options = []
    original_mapping = {} 


    for key, value in options.items():
        # aplit and rejoin
        reformatted = ', '.join(value.split(';'))
        all_options.append(reformatted) 
        original_mapping[reformatted] = key  

    # display options
    selected_reformatted = st.selectbox(f'{featureNameChange(feature_name)}', list(all_options))

    # mapping the relavant key
    selected_key = original_mapping[selected_reformatted]

    # add the key
    features.append(int(selected_key))


# numerical inputs
years_code_pro = st.number_input('Years of Professional Coding Experience (1-28)', min_value=1, max_value=28)
work_experience = st.number_input('Years of Work Experience (0-30)', min_value=0, max_value=30)

# encode the values
encoded_years_code_pro = yearsCodeProEncoder(years_code_pro)
encoded_work_experience = workExpEncoder(work_experience)

# add encoded
features.append(encoded_years_code_pro)
features.append(encoded_work_experience)

# ready to predict
features = np.array([features])

if st.button('Predict', type="primary"):
    prediction = best_model.predict(features)
    st.success(f"You are eligible for: {prediction[0]} USD")
