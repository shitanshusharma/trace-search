"use strict";

(function () {
  var root = document.querySelector("main");
  if (!root) {
    return;
  }
  root.setAttribute("data-ready", "true");

  var form = document.querySelector("form");
  if (form) {
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      sendSignIn();
    });
  }

  var signOut = document.getElementById("signout");
  if (signOut) {
    signOut.addEventListener("click", function () {
      sendSignOut();
    });
  }

  function fieldValue(id) {
    var element = document.getElementById(id);
    if (!element) {
      return "";
    }
    return element.value;
  }

  function sendSignIn() {
    var payload = {
      username: fieldValue("username"),
      password: fieldValue("password")
    };
    fetch("/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify(payload)
    }).then(function (response) {
      return response.json().then(function (body) {
        var status = document.getElementById("status");
        if (response.ok && body.csrf_token) {
          if (status) {
            status.textContent = "signed in";
          }
          var page = document.querySelector("main");
          if (page) {
            page.setAttribute("data-csrf", body.csrf_token);
          }
          return;
        }
        if (status) {
          status.textContent = "sign-in refused";
        }
      });
    }).catch(function () {
      var status = document.getElementById("status");
      if (status) {
        status.textContent = "sign-in refused";
      }
    });
  }

  function sendSignOut() {
    var page = document.querySelector("main");
    var csrf = page ? page.getAttribute("data-csrf") : "";
    fetch("/logout", {
      method: "POST",
      credentials: "same-origin",
      headers: { "X-CSRF-Token": csrf || "" }
    }).then(function (response) {
      var status = document.getElementById("status");
      if (!status) {
        return;
      }
      status.textContent = response.ok ? "signed out" : "sign-out refused";
    });
  }
})();
