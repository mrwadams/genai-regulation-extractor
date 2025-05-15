# Define multipage apps with `st.Page` and `st.navigation`

`st.Page` and `st.navigation` are the preferred commands for defining multipage apps. With these commands, you have flexibility to organize your project files and customize your navigation menu. Simply initialize `StreamlitPage` objects with `st.Page`, then pass those `StreamlitPage` objects to `st.navigation` in your entrypoint file (i.e. the file you pass to `streamlit run`).

## App structure

When using `st.navigation`, your entrypoint file acts like a page router. Each page is a script executed from your entrypoint file. You can define a page from a Python file or function. If you include elements or widgets in your entrypoint file, they become common elements between your pages. In this case, you can think of your entrypoint file like a picture frame around each of your pages.

You can only call `st.navigation` once per app run and you must call it from your entrypoint file. When a user selects a page in navigation (or is routed through a command like `st.switch_page`), `st.navigation` returns the selected page. You must manually execute that page with the `.run()` method.

## Defining pages

`st.Page` lets you define a page. The first and only required argument defines your page source, which can be a Python file or function. When using Python files, your pages may be in a subdirectory (or superdirectory). The path to your page file must always be relative to the entrypoint file. Once you create your page objects, pass them to `st.navigation` to register them as pages in your app.

Within `st.Page`, Streamlit uses `title` to set the page label and title. Additionally, Streamlit uses `icon` to set the page icon and favicon. If you want to have a different page title and label, or different page icon and favicon, you can use `st.set_page_config` to change the page title and/or favicon. Just call `st.set_page_config` after `st.navigation`, either in your entrypoint file or in your page source.

## Customizing navigation

If you want to group your pages into sections, `st.navigation` lets you insert headers within your navigation. Alternatively, you can disable the default navigation widget and build a custom navigation menu with `st.page_link`.

Additionally, you can dynamically change which pages you pass to `st.navigation`. However, only the page returned by `st.navigation` accepts the `.run()` method. If a user enters a URL with a pathname, and that pathname is not associated to a page in `st.navigation` (on first run), Streamlit will throw a "Page not found" error and redirect them to the default page.

### Adding section headers

As long as you don't want to hide a valid, accessible page in the navigation menu, the simplest way to customize your navigation menu is to organize the pages within `st.navigation`. You can sort or group pages, as well as remove any pages you don't want the user to access. This is a convenient way to handle user permissions.

### Dynamically changing the available pages

You can change what pages are available to a user by updating the list of pages in `st.navigation`. This is a convenient way to handle role-based or user-based access to certain pages.

### Building a custom navigation menu

If you want more control over your navigation menu, you can hide the default navigation and build your own. You can hide the default navigation by including `position="hidden"` in your `st.navigation` command. If you want a page to be available to a user without showing it in the navigation menu, you must use this method. A user can't be routed to a page if the page isn't included in `st.navigation`. This applies to navigation by URL as well as commands like `st.switch_page` and `st.page_link`. 